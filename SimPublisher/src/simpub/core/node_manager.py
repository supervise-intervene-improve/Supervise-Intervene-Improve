from __future__ import annotations

import asyncio
import concurrent.futures
import socket
import struct
import time
import traceback
from asyncio import sleep as async_sleep
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from json import loads
from typing import Callable, Optional

import zmq
import zmq.asyncio

from ..simpubweb.simpub_web_server import SimPubWebServer
from .log import logger
from .utils import (
    DISCOVERY_PORT,
    MULTICAST_GRP,
    XRNodeInfo,
    XRNodeRegistry,
    send_string_request_async,
)


# NOTE: asyncio.loop.sock_recvfrom can only be used after Python 3.11
# So we create a custom DatagramProtocol for multicast discovery
class MulticastDiscoveryProtocol(asyncio.DatagramProtocol):
    """DatagramProtocol for handling multicast discovery messages"""

    def __init__(self, node_manager: XRNodeManager):
        self.node_manager = node_manager
        self.registry = node_manager.xr_nodes
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport
        logger.info("Multicast discovery connection established")

    def datagram_received(self, data: bytes, addr: tuple[str, int]):
        """Handle incoming multicast discovery messages"""
        try:
            node_ip, message = addr[0], data.decode("utf-8")
            node_id, node_info_id, service_port = (
                message[:36],
                message[36:72],
                message[72:],
            )
            entry = self.registry.get(node_id)
            if (
                entry is None
                or entry.info is None
                or entry.info["nodeInfoID"] != node_info_id
            ):
                # Schedule the async registration when node info is unknown
                self.node_manager.submit_asyncio_task(
                    self.node_manager.register_node_info_async,
                    node_id,
                    node_ip,
                    service_port,
                )
            self.registry.touch(node_id)
        except Exception as e:
            logger.error(f"Error processing datagram: {e}")
            traceback.print_exc()

    def error_received(self, exc):
        logger.error(f"Multicast protocol error: {exc}")

    def connection_lost(self, exc):
        if exc:
            logger.error(f"Multicast connection lost: {exc}")
        else:
            logger.error("Multicast discovery connection closed")


@dataclass(frozen=True)
class XRTargetSelection:
    status: str
    requested_name: Optional[str]
    selected_id: Optional[str] = None
    selected_info: Optional[XRNodeInfo] = None
    candidate_summaries: tuple[str, ...] = ()
    message: str = ""


class XRNodeManager:
    manager: Optional[XRNodeManager] = None

    def __init__(self, host_ip: str) -> None:
        XRNodeManager.manager = self
        self.zmq_context = zmq.asyncio.Context.instance()
        self.host_ip: str = host_ip
        self.xr_nodes = XRNodeRegistry()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.server_future = self.executor.submit(self.thread_task)
        self.web_server: Optional[SimPubWebServer] = None
        self.web_server_future: Optional[concurrent.futures.Future] = None
        self._start_web_server()
        self.discovery_transport = None  # Track the transport for cleanup
        # Wait for the loop
        while not hasattr(self, "loop"):
            time.sleep(0.01)

    def start_discover_node_loop(self):
        """Start the async discovery loop in the event loop"""
        if self.loop and self.loop.is_running():
            # Submit the async task to the event loop
            self.discovery_task = self.submit_asyncio_task(
                self.xr_node_discover_loop
            )
            self.node_heartbeat_task = self.submit_asyncio_task(
                self.check_node_heartbeat
            )
        else:
            logger.error(
                "Event loop is not running, cannot start discovery loop"
            )

    def _start_web_server(self):
        if self.web_server is not None:
            return
        try:
            self.web_server = SimPubWebServer(
                self.xr_nodes, host="127.0.0.1", port=5000
            )
            self.web_server_future = self.executor.submit(
                self.web_server.serve_forever
            )
            logger.info("Web dashboard is running at http://127.0.0.1:5000")
        except Exception as e:
            logger.error(
                f"Failed to start web dashboard on 127.0.0.1:5000: {e}"
            )
            traceback.print_exc()

    def _stop_web_server(self):
        if self.web_server is None:
            return
        try:
            self.web_server.shutdown()
            if self.web_server_future is not None:
                try:
                    self.web_server_future.result(timeout=2)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Timed out while waiting for web dashboard shutdown"
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.error(
                        "Error waiting for web dashboard to stop: %s",
                        e,
                    )
        except Exception as e:
            logger.error("Error during web dashboard shutdown: %s", e)
            traceback.print_exc()
        finally:
            self.web_server = None
            self.web_server_future = None

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def thread_task(self):
        logger.info("The node is running...")
        try:
            self.start_event_loop()
        except KeyboardInterrupt:
            self.stop_node()
        except Exception as e:
            logger.error(f"Unexpected error in thread_task: {e}")
        finally:
            logger.info("The node has been stopped")

    def stop_node(self):
        logger.info("Start to stop the node")
        self.running = False
        self._stop_web_server()
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.stop_tasks)
                time.sleep(0.1)
                self.loop.stop()
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        logger.info("Start to shutdown the executor")
        self.executor.shutdown(wait=False)
        logger.info("The executor has been shutdown")
        XRNodeManager.manager = None

    def stop_tasks(self):
        # Close discovery transport if it exists
        if self.discovery_transport:
            self.discovery_transport.close()

        # Cancel all running tasks including discovery
        for task in asyncio.all_tasks():
            if task is asyncio.current_task():
                continue
            task.cancel()
        logger.info("All tasks have been cancelled")

    @staticmethod
    def _service_port_from_info(info: XRNodeInfo) -> Optional[int]:
        port = (
            info.get("port")
            or info.get("servicePort")
            or info.get("service_port")
            or info.get("rpcPort")
            or info.get("rpc_port")
        )
        if port is None:
            return None
        return int(port)

    @staticmethod
    def describe_xr_node(info: XRNodeInfo) -> str:
        port = XRNodeManager._service_port_from_info(info)
        port_str = "?" if port is None else str(port)
        node_type = info.get("type", "?")
        name = info.get("name", "<unnamed>")
        ip = info.get("ip", "?")
        return f"{name}@{ip}:{port_str} type={node_type}"

    @staticmethod
    def _is_unity_xr_candidate(info: Optional[XRNodeInfo]) -> bool:
        if info is None:
            return False
        if info.get("type") == "SimPub" or info.get("name") == "SimPub":
            return False
        if info.get("type") == "UnityNode":
            return True
        topic_dict = info.get("topicDict") or {}
        return (
            "MotionController" in topic_dict
            or "HandTracking" in topic_dict
            or "ConsoleLogger" in topic_dict
        )

    def list_unity_xr_candidates(self) -> list[tuple[str, XRNodeInfo]]:
        candidates: list[tuple[str, XRNodeInfo]] = []
        for node_id, entry in list(self.xr_nodes.items()):
            info = entry.info
            if not self._is_unity_xr_candidate(info):
                continue
            candidates.append((node_id, info))
        return candidates

    def get_node_entry(self, node_id: str):
        return self.xr_nodes.get(node_id)

    def get_node_info(self, node_id: str) -> Optional[XRNodeInfo]:
        entry = self.get_node_entry(node_id)
        if entry is None:
            return None
        return entry.info

    def is_node_stable(
        self,
        node_id: str,
        *,
        min_age_seconds: float,
        min_heartbeats: int = 2,
    ) -> bool:
        entry = self.get_node_entry(node_id)
        if entry is None or entry.info is None:
            return False
        return (
            entry.age_seconds() >= max(0.0, min_age_seconds)
            and entry.heartbeat_count >= max(1, min_heartbeats)
        )

    def resolve_preferred_xr_target(
        self, preferred_name: Optional[str]
    ) -> XRTargetSelection:
        candidates = self.list_unity_xr_candidates()
        candidate_summaries = tuple(
            self.describe_xr_node(info) for _, info in candidates
        )
        requested_name = preferred_name or None

        if requested_name:
            exact_matches = [
                (node_id, info)
                for node_id, info in candidates
                if info.get("name") == requested_name
            ]
            if len(exact_matches) == 1:
                node_id, info = exact_matches[0]
                return XRTargetSelection(
                    status="exact",
                    requested_name=requested_name,
                    selected_id=node_id,
                    selected_info=info,
                    candidate_summaries=candidate_summaries,
                    message=(
                        f"Resolved requested XR target '{requested_name}' to "
                        f"{self.describe_xr_node(info)}."
                    ),
                )
            if len(exact_matches) > 1:
                return XRTargetSelection(
                    status="ambiguous",
                    requested_name=requested_name,
                    candidate_summaries=candidate_summaries,
                    message=(
                        f"Multiple XR nodes matched '{requested_name}': "
                        + ", ".join(
                            self.describe_xr_node(info)
                            for _, info in exact_matches
                        )
                    ),
                )
            if len(candidates) == 1:
                node_id, info = candidates[0]
                return XRTargetSelection(
                    status="auto_single",
                    requested_name=requested_name,
                    selected_id=node_id,
                    selected_info=info,
                    candidate_summaries=candidate_summaries,
                    message=(
                        f"Requested XR node '{requested_name}' was not found. "
                        f"Auto-selecting the only available XR node "
                        f"{self.describe_xr_node(info)}."
                    ),
                )
            if len(candidates) == 0:
                return XRTargetSelection(
                    status="unavailable",
                    requested_name=requested_name,
                    candidate_summaries=(),
                    message=(
                        f"Waiting for XR node '{requested_name}'. "
                        "No Unity XR nodes are currently registered."
                    ),
                )
            return XRTargetSelection(
                status="ambiguous",
                requested_name=requested_name,
                candidate_summaries=candidate_summaries,
                message=(
                    f"Requested XR node '{requested_name}' was not found and "
                    "multiple Unity XR nodes are available: "
                    + ", ".join(candidate_summaries)
                ),
            )

        if len(candidates) == 1:
            node_id, info = candidates[0]
            return XRTargetSelection(
                status="single",
                requested_name=None,
                selected_id=node_id,
                selected_info=info,
                candidate_summaries=candidate_summaries,
                message=f"Using the only available XR node {self.describe_xr_node(info)}.",
            )
        if len(candidates) == 0:
            return XRTargetSelection(
                status="unavailable",
                requested_name=None,
                candidate_summaries=(),
                message="No Unity XR nodes are currently registered.",
            )
        return XRTargetSelection(
            status="multiple",
            requested_name=None,
            candidate_summaries=candidate_summaries,
            message="Multiple Unity XR nodes are available: " + ", ".join(candidate_summaries),
        )

    def spin(self):
        while True:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
        self.stop_node()
        logger.info("The node has been stopped")

    def submit_asyncio_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            raise RuntimeError("The event loop is not running")
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def xr_node_discover_loop(self):
        # Create multicast socket
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        # Allow reuse of address
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the port
        sock.bind(("", DISCOVERY_PORT))
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(MULTICAST_GRP),
            socket.inet_aton(self.host_ip),
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        logger.info(
            f"Listening for multicast on {MULTICAST_GRP}:{DISCOVERY_PORT}"
            f" from {self.host_ip}"
        )
        # Get event loop and create datagram endpoint
        loop = asyncio.get_event_loop()
        try:
            # Create the datagram endpoint with the protocol
            transport, _ = await loop.create_datagram_endpoint(
                lambda: MulticastDiscoveryProtocol(self), sock=sock
            )
            # Store transport for cleanup
            self.discovery_transport = transport
            # Keep the loop running until stopped
            while self.running:
                await async_sleep(1)  # Keep the coroutine alive
        except asyncio.CancelledError:
            logger.info("Discovery loop cancelled...")
        except Exception as e:
            logger.error(f"Error in discovery loop: {e}")
            traceback.print_exc()
        finally:
            # Clean up
            if "transport" in locals():
                transport.close()
            sock.close()
            logger.info("Multicast discovery loop stopped")

    async def register_node_info_async(
        self, node_id: str, node_ip: str, service_port: str
    ) -> None:
        try:
            node_info_bytes = await send_string_request_async(
                ["GetNodeInfo", ""], f"tcp://{node_ip}:{service_port}"
            )
            if node_info_bytes is None:
                logger.error(
                    f"Failed to get node info from {node_ip}:{service_port}"
                )
                return
            node_info: XRNodeInfo = loads(node_info_bytes.decode("utf-8"))
            raw = node_info_bytes.decode("utf-8", errors="replace")
            node_info = loads(raw)

            # Some nodes return JSON as a *string containing JSON*.
            if isinstance(node_info, str):
                node_info = loads(node_info)

            if not isinstance(node_info, dict):
                raise TypeError(f"Expected node_info dict, got {type(node_info)}: {node_info!r}")

            # Normalize keys so downstream code doesn't KeyError on 'port'
            node_info.setdefault("port", node_info.get("servicePort") or node_info.get("topicPort"))

            node_info["ip"] = node_ip
            self.xr_nodes.update_info(node_id, node_info)
            logger.info(
                f"Node: {node_info['name']}/{node_id} "
                f"at {node_ip}:{service_port} registered successfully"
            )
            logger.info(
                "XR node details: %s services=%d topics=%d",
                self.describe_xr_node(node_info),
                len(node_info.get("serviceList", [])),
                len(node_info.get("topicDict", {})),
            )
        except Exception as e:
            logger.error(f"Error in register_node_info: {e}")
            traceback.print_exc()

    async def check_node_heartbeat(self):
        """Check the heartbeat of registered nodes and remove offline nodes."""
        while self.running:
            offline_nodes = self.xr_nodes.remove_offline(timeout=5.0)
            for node_id, entry in offline_nodes:
                node_name = entry.info["name"] if entry.info else node_id
                logger.warning(
                    f"Node {node_name} {node_id} is offline for 5s,"
                    " removing it"
                )
            await async_sleep(1)


def init_xr_node_manager(ip_addr: Optional[str] = None) -> XRNodeManager:
    if XRNodeManager.manager is not None:
        # Idempotent on same IP: allow callers to re-acquire the singleton
        # by passing either None (no opinion) or the same IP they originally
        # initialized with. Only raise if the caller is asking for a
        # different IP — that genuinely indicates a bug.
        if ip_addr is None or ip_addr == XRNodeManager.manager.host_ip:
            return XRNodeManager.manager
        raise RuntimeError(
            f"XRNodeManager already initialized with IP "
            f"{XRNodeManager.manager.host_ip}, cannot reinitialize "
            f"with different IP {ip_addr}."
        )
    if ip_addr is None:
        raise ValueError(
            "IP address must be provided for the first initialization"
        )
    logger.info(f"Initializing XRNodeManager with IP {ip_addr}")
    return XRNodeManager(ip_addr)
