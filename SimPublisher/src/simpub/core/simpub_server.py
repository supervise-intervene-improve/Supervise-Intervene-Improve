from __future__ import annotations

import abc
import time
import traceback
from asyncio import sleep as asyncio_sleep
from typing import Dict, List, Optional

from ..parser.simdata import SimObject, SimScene
from .log import func_timing, logger
from .net_component import Streamer
from .node_manager import XRNodeManager, XRTargetSelection, init_xr_node_manager
from .utils import (
    XRNodeInfo,
    get_zmq_socket_url,
    send_request_with_addr_async,
)


def _xr_service_addr(xr_info: dict) -> str:
    """
    Build the service (RPC) address for an XR node.

    Different IRXR builds may expose the RPC port under different keys.
    """
    port = (
        xr_info.get("port")
        or xr_info.get("servicePort")
        or xr_info.get("service_port")
        or xr_info.get("rpcPort")
        or xr_info.get("rpc_port")
    )
    if not port:
        raise KeyError(
            f"XR node info has no service port. keys={list(xr_info.keys())} xr_info={xr_info}"
        )
    return f"tcp://{xr_info['ip']}:{int(port)}"


class ServerBase(abc.ABC):
    def __init__(self, ip_addr: str):
        self.ip_addr: str = ip_addr
        self.node_manager = init_xr_node_manager(ip_addr)
        self.initialize()
        self.node_manager.start_discover_node_loop()

    def spin(self):
        self.node_manager.spin()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    def shutdown(self):
        self.node_manager.stop_node()


class MsgServer(ServerBase):
    def initialize(self):
        pass


class SimPublisher(ServerBase):
    SCENE_RPC_TIMEOUT_SECONDS = 5
    ASSET_RPC_TIMEOUT_SECONDS = 15
    SCENE_RPC_RETRIES = 3
    SCENE_SPAWN_GRACE_SECONDS = 1.0
    XR_NODE_STABILITY_WINDOW_SECONDS = 2.0
    XR_NODE_MIN_HEARTBEATS = 2
    XR_SEARCH_POLL_INTERVAL_SECONDS = 0.5

    def __init__(
        self,
        sim_scene: SimScene,
        ip_addr: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
        fps: int = 45,
        preferred_xr_name: Optional[str] = None,
    ) -> None:
        self.sim_scene = sim_scene
        self.fps = fps
        self.preferred_xr_name = preferred_xr_name
        self._last_target_selection_signature = None
        self._last_transfer_suppression_reason = None
        self._last_transfer_state_signature = None
        self._pinned_xr_node_id: Optional[str] = None
        self._scene_transfer_in_flight = False
        self._scene_transfer_completed = False
        self._scene_transfer_failed = False
        self._retry_same_node_after = 0.0
        if no_rendered_objects is None:
            self.no_rendered_objects = []
        else:
            self.no_rendered_objects = no_rendered_objects
        if no_tracked_objects is None:
            self.no_tracked_objects = []
        else:
            self.no_tracked_objects = no_tracked_objects
        super().__init__(ip_addr)

    def initialize(self) -> None:
        self.scene_update_streamer = Streamer(
            topic_name="RigidObjectUpdate",
            update_func=self.get_update,
            fps=self.fps,
            start_streaming=True,
        )
        self.node_manager.submit_asyncio_task(
            self.search_xr_device, self.node_manager
        )

    def shutdown_scene_only(self) -> None:
        """Stop only the scene update streamer. Leaves the shared
        XRNodeManager + any other services on this node alive so the
        surrounding process (sensor publisher, MetaQuest3 listener) keeps
        working. Pair with PROMOTE/DEMOTE in multi-session selector flow."""
        try:
            streamer = getattr(self, "scene_update_streamer", None)
            if streamer is not None:
                streamer.shutdown()
        except Exception:
            pass

    async def search_xr_device(self, node_manager: XRNodeManager):
        while node_manager.running:
            try:
                pinned_info = self._get_pinned_target_info(node_manager)
                if pinned_info is None:
                    selection = node_manager.resolve_preferred_xr_target(
                        self.preferred_xr_name
                    )
                    self._log_target_selection(selection, node_manager)
                    if (
                        selection.selected_id is None
                        or selection.selected_info is None
                    ):
                        self._log_transfer_suppressed(
                            "waiting for a selectable XR target"
                        )
                        await asyncio_sleep(
                            self.XR_SEARCH_POLL_INTERVAL_SECONDS
                        )
                        continue
                    self._pin_target(
                        selection.selected_id,
                        selection.selected_info,
                        selection.status,
                        node_manager,
                    )
                    pinned_info = self._get_pinned_target_info(node_manager)

                if pinned_info is not None:
                    pinned_node_id, xr_info = pinned_info
                    await self._maybe_send_scene_to_pinned_target(
                        node_manager,
                        pinned_node_id,
                        xr_info,
                    )
            except Exception as e:
                logger.error(f"Error when sending scene to xr device: {e}")
                traceback.print_exc()
            await asyncio_sleep(self.XR_SEARCH_POLL_INTERVAL_SECONDS)

    def _get_pinned_target_info(
        self,
        node_manager: XRNodeManager,
    ) -> Optional[tuple[str, XRNodeInfo]]:
        if self._pinned_xr_node_id is None:
            return None
        xr_info = node_manager.get_node_info(self._pinned_xr_node_id)
        if xr_info is None:
            self._unpin_target(
                "pinned XR target went offline or stopped responding"
            )
            return None
        return self._pinned_xr_node_id, xr_info

    def _pin_target(
        self,
        node_id: str,
        xr_info: XRNodeInfo,
        selection_status: str,
        node_manager: XRNodeManager,
    ) -> None:
        if self._pinned_xr_node_id == node_id:
            return
        if self._pinned_xr_node_id is not None:
            self._unpin_target(
                "replacing pinned XR target with a newly selected node"
            )
        self._pinned_xr_node_id = node_id
        self._scene_transfer_in_flight = False
        self._scene_transfer_completed = False
        self._scene_transfer_failed = False
        self._retry_same_node_after = 0.0
        self._last_transfer_state_signature = None
        self._clear_transfer_suppression()
        logger.info(
            "Pinned XR target for scene '%s' to %s (selection=%s).",
            self.sim_scene.name,
            node_manager.describe_xr_node(xr_info),
            selection_status,
        )

    def _unpin_target(self, reason: str) -> None:
        if self._pinned_xr_node_id is None:
            return
        logger.warning(
            "Unpinned XR target for scene '%s' (%s).",
            self.sim_scene.name,
            reason,
        )
        self._pinned_xr_node_id = None
        self._scene_transfer_in_flight = False
        self._scene_transfer_completed = False
        self._scene_transfer_failed = False
        self._retry_same_node_after = 0.0
        self._last_transfer_state_signature = None
        self._clear_transfer_suppression()

    async def _maybe_send_scene_to_pinned_target(
        self,
        node_manager: XRNodeManager,
        node_id: str,
        xr_info: XRNodeInfo,
    ) -> None:
        target_desc = node_manager.describe_xr_node(xr_info)
        if not node_manager.is_node_stable(
            node_id,
            min_age_seconds=self.XR_NODE_STABILITY_WINDOW_SECONDS,
            min_heartbeats=self.XR_NODE_MIN_HEARTBEATS,
        ):
            self._log_transfer_suppressed(
                f"waiting for pinned XR target to become stable: {target_desc}"
            )
            return

        if self._scene_transfer_in_flight:
            self._log_transfer_suppressed(
                f"scene transfer already in flight for pinned target: {target_desc}"
            )
            return

        if self._scene_transfer_completed and not self._scene_transfer_failed:
            self._log_transfer_suppressed(
                f"scene already transferred to pinned target: {target_desc}"
            )
            return

        if self._scene_transfer_failed:
            if time.monotonic() < self._retry_same_node_after:
                self._log_transfer_suppressed(
                    f"waiting before retrying failed transfer to pinned target: {target_desc}"
                )
                return
            logger.info(
                "Retrying scene transfer for '%s' on pinned XR target %s.",
                self.sim_scene.name,
                target_desc,
            )

        await self.send_scene_to_xr_device(xr_info, pinned_node_id=node_id)

    def _clear_transfer_suppression(self) -> None:
        self._last_transfer_suppression_reason = None

    def _log_transfer_suppressed(self, reason: str) -> None:
        if reason == self._last_transfer_suppression_reason:
            return
        self._last_transfer_suppression_reason = reason
        logger.info(
            "Suppressed scene resend for '%s': %s",
            self.sim_scene.name,
            reason,
        )

    def _log_transfer_state(
        self,
        state: str,
        target_desc: str,
        detail: str = "",
    ) -> None:
        signature = (state, target_desc, detail)
        if signature == self._last_transfer_state_signature:
            return
        self._last_transfer_state_signature = signature
        suffix = f" ({detail})" if detail else ""
        if state == "failed":
            logger.warning(
                "Scene transfer state for '%s': %s on %s%s",
                self.sim_scene.name,
                state,
                target_desc,
                suffix,
            )
        else:
            logger.info(
                "Scene transfer state for '%s': %s on %s%s",
                self.sim_scene.name,
                state,
                target_desc,
                suffix,
            )

    def _log_target_selection(
        self, selection: XRTargetSelection, node_manager: XRNodeManager
    ) -> None:
        signature = (
            selection.status,
            selection.requested_name,
            selection.selected_id,
            selection.candidate_summaries,
        )
        if signature == self._last_target_selection_signature:
            return
        self._last_target_selection_signature = signature

        if selection.status == "exact":
            logger.info(selection.message)
        elif selection.status == "auto_single":
            logger.warning(selection.message)
        elif selection.status == "unavailable":
            logger.warning(selection.message)
        elif selection.status in {"ambiguous", "multiple"}:
            logger.error(selection.message)
        else:
            logger.info(selection.message)

        if selection.selected_info is not None:
            logger.info(
                "Scene publisher target for '%s': %s",
                self.sim_scene.name,
                node_manager.describe_xr_node(selection.selected_info),
            )

    @func_timing
    async def send_scene_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        *,
        pinned_node_id: Optional[str] = None,
    ):
        if self._scene_transfer_in_flight:
            raise RuntimeError(
                f"Scene transfer for '{self.sim_scene.name}' is already in flight."
            )

        target_desc = XRNodeManager.describe_xr_node(xr_info)
        self._scene_transfer_in_flight = True
        self._scene_transfer_completed = False
        self._scene_transfer_failed = False
        self._retry_same_node_after = 0.0
        self._clear_transfer_suppression()
        self._log_transfer_state("starting", target_desc)
        try:
            logger.info(
                "Sending scene '%s' to xr device %s",
                self.sim_scene.name,
                target_desc,
            )
            await self._send_request_or_raise(
                [
                    "DeleteSimScene".encode(),
                    self.sim_scene.name.encode(),
                ],
                _xr_service_addr(xr_info),
                op_name="DeleteSimScene",
            )
            logger.info(
                "DeleteSimScene acknowledged for scene '%s' by %s",
                self.sim_scene.name,
                target_desc,
            )
            await self._send_request_or_raise(
                [
                    "SpawnSimScene".encode(),
                    self.sim_scene.serialize().encode(),
                ],
                _xr_service_addr(xr_info),
                op_name="SpawnSimScene",
            )
            logger.info(
                "SpawnSimScene acknowledged for scene '%s' by %s",
                self.sim_scene.name,
                target_desc,
            )
            await asyncio_sleep(self.SCENE_SPAWN_GRACE_SECONDS)
            if self.sim_scene.root is None:
                logger.warning("The SimScene root is None, nothing to send.")
                self._scene_transfer_completed = True
                self._log_transfer_state(
                    "completed",
                    target_desc,
                    "scene-root-none",
                )
                return
            await self.send_rigid_body_streamer(
                xr_info,
                self.sim_scene,
            )
            await self.send_objects_to_xr_device(
                xr_info,
                self.sim_scene,
                self.sim_scene.root,
            )
            await self.send_assets_to_xr_device(
                xr_info,
                self.sim_scene,
            )
            self._scene_transfer_completed = True
            logger.info(
                "Finished scene transfer for '%s' to %s",
                self.sim_scene.name,
                target_desc,
            )
            self._log_transfer_state("completed", target_desc)
        except Exception:
            self._scene_transfer_failed = True
            self._scene_transfer_completed = False
            self._retry_same_node_after = (
                time.monotonic() + self.XR_NODE_STABILITY_WINDOW_SECONDS
            )
            detail = (
                f"retrying pinned-node transfer after "
                f"{self.XR_NODE_STABILITY_WINDOW_SECONDS:.1f}s"
            )
            if (
                pinned_node_id is not None
                and self._pinned_xr_node_id is not None
                and pinned_node_id != self._pinned_xr_node_id
            ):
                detail += "; pinned target changed"
            self._log_transfer_state("failed", target_desc, detail)
            raise
        finally:
            self._scene_transfer_in_flight = False

    async def send_objects_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
        sim_object: SimObject,
        parent: Optional[SimObject] = None,
    ):
        await self._send_request_or_raise(
            [
                f"{sim_scene.name}/CreateSimObject".encode(),
                parent.name.encode() if parent else "".encode(),
                sim_object.serialize().encode(),
            ],
            _xr_service_addr(xr_info),
            op_name=f"{sim_scene.name}/CreateSimObject:{sim_object.name}",
        )
        logger.info(
            "CreateSimObject succeeded for '%s' in scene '%s' on %s",
            sim_object.name,
            sim_scene.name,
            XRNodeManager.describe_xr_node(xr_info),
        )

    async def send_assets_to_xr_device(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
    ):
        if sim_scene.root is None:
            logger.warning("The SimScene root is None, nothing to send.")
            return
        asset_count = 0
        for (
            name,
            sim_visual,
            mesh_raw_data,
            texture_raw_data,
        ) in sim_scene.get_all_assets(sim_scene.root):
            visual_name = sim_visual.name or "<unnamed>"
            await self._send_request_or_raise(
                [
                    f"{sim_scene.name}/CreateVisual".encode(),
                    name.encode(),
                    sim_visual.serialize().encode(),
                    mesh_raw_data,
                    texture_raw_data,
                ],
                _xr_service_addr(xr_info),
                timeout=self.ASSET_RPC_TIMEOUT_SECONDS,
                op_name=f"{sim_scene.name}/CreateVisual:{name}:{visual_name}",
            )
            asset_count += 1
            logger.info(
                "CreateVisual succeeded for obj='%s' visual='%s' in scene '%s' on %s",
                name,
                visual_name,
                sim_scene.name,
                XRNodeManager.describe_xr_node(xr_info),
            )
        logger.info(
            "CreateVisual succeeded for %d asset(s) in scene '%s' to %s",
            asset_count,
            sim_scene.name,
            XRNodeManager.describe_xr_node(xr_info),
        )

    async def send_rigid_body_streamer(
        self,
        xr_info: XRNodeInfo,
        sim_scene: SimScene,
    ):
        url = get_zmq_socket_url(self.scene_update_streamer.socket)
        await self._send_request_or_raise(
            [
                f"{sim_scene.name}/SubscribeRigidObjectsController".encode(),
                url.encode(),
                "RigidObjectUpdate".encode(),
            ],
            _xr_service_addr(xr_info),
            op_name=f"{sim_scene.name}/SubscribeRigidObjectsController",
        )
        logger.info(
            "SubscribeRigidObjectsController acknowledged for scene '%s' by %s using %s",
            sim_scene.name,
            XRNodeManager.describe_xr_node(xr_info),
            url,
        )

    async def _send_request_or_raise(
        self,
        messages: List[bytes],
        addr: str,
        *,
        op_name: str,
        timeout: int | None = None,
        retries: int | None = None,
    ) -> bytes:
        timeout = (
            self.SCENE_RPC_TIMEOUT_SECONDS if timeout is None else timeout
        )
        retries = self.SCENE_RPC_RETRIES if retries is None else retries
        last_result = None

        for attempt in range(1, max(1, retries) + 1):
            last_result = await send_request_with_addr_async(
                messages,
                addr,
                timeout=timeout,
            )
            if last_result is not None:
                logger.debug(f"{op_name} succeeded for {addr}")
                if attempt > 1:
                    logger.warning(
                        f"{op_name} succeeded on retry {attempt}/{retries} for {addr}"
                    )
                return last_result

            logger.warning(
                f"{op_name} timed out on attempt {attempt}/{retries} for {addr}"
            )
            if attempt < retries:
                await asyncio_sleep(min(0.25 * attempt, 1.0))

        raise TimeoutError(
            f"{op_name} timed out after {retries} attempt(s) for {addr}. "
            f"last_result={last_result!r}"
        )

    def _on_asset_request(self, req: str) -> bytes:
        return self.sim_scene.raw_data[req]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
