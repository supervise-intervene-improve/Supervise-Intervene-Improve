import json
import time
import traceback
from asyncio import sleep as async_sleep
from typing import Optional

import zmq

from ..core.log import logger
from ..core.net_component import Subscriber
from ..core.node_manager import (
    XRNodeInfo,
    XRTargetSelection,
    XRNodeManager,
    init_xr_node_manager,
)
from ..core.utils import AsyncSocket, print_node_info, send_request_async


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDevice:
    type = "XRDevice"

    def __init__(
        self,
        device_name: str = "UnityNode",
    ) -> None:
        self.manager = init_xr_node_manager()
        self.running = True
        self.connected = False
        self.device_name = device_name
        self.device_id: Optional[str] = None
        self.device_info: Optional[XRNodeInfo] = None
        self.connected_node_name: Optional[str] = None
        self._last_target_selection_signature = None
        self.req_socket: AsyncSocket = self.manager.create_socket(zmq.REQ)
        self.sub_list: list[Subscriber] = []
        self.sub_list.append(Subscriber("ConsoleLogger", self.print_log))
        self.manager.submit_asyncio_task(self.checking_connection)

    def wait_for_connection(self):
        """
        Wait for the connection to the XR device.
        This method blocks until the connection is established.
        """
        while not self.connected:
            time.sleep(0.1)

    async def checking_connection(self):
        logger.info(f"checking the connection to {self.device_name}")
        while self.running:
            selection = self.manager.resolve_preferred_xr_target(
                self.device_name
            )
            self._log_target_selection(selection)
            node_info = selection.selected_info
            if node_info is None or selection.selected_id is None:
                await async_sleep(0.5)
                continue
            if node_info["nodeID"] == self.device_id:
                await async_sleep(0.5)
                continue
            if self.device_info is not None:
                self.disconnect()
            self.device_info = node_info
            self.device_id = node_info["nodeID"]
            self.connected_node_name = node_info.get("name")
            self.connected = True
            self.subscribe_to_client(node_info)
            print_node_info(node_info)
            await async_sleep(0.5)
        # if self.device_info is None:
        #     return
        # self.connect_to_client(self.device_info)

    def _log_target_selection(self, selection: XRTargetSelection):
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
                "XRDevice '%s' will connect to %s",
                self.device_name,
                XRNodeManager.describe_xr_node(selection.selected_info),
            )

    def subscribe_to_client(self, info: XRNodeInfo):
        try:
            self.req_socket.connect(f"tcp://{info['ip']}:{info['port']}")
            for sub in self.sub_list:
                if sub.topic_name not in info["topicDict"]:
                    continue
                sub.start_connection(
                    f"tcp://{info['ip']}:{info['topicDict'][sub.topic_name]}"
                )
                logger.info(
                    f"Subscribed to {sub.topic_name} "
                    f"on {info['ip']}:{info['port']}"
                )
            logger.remote_log(
                f"{self.type} Connected to requested '{self.device_name}' "
                f"via node '{info['name']}' at {info['ip']}:{info['port']}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to {self.device_name} at "
                f"{info['ip']}:{info['port']}: {e}"
            )
            traceback.print_exc()
            return

    def request(self, service_name: str, request: str) -> str:
        if self.device_info is None:
            logger.error(f"Device {self.device_name} is not connected")
            return ""
        if service_name not in self.device_info["serviceList"]:
            logger.error(f'"{service_name}" Service is not available')
            return ""
        messages = [
            service_name.encode(),
            request.encode(),
        ]
        future = self.manager.submit_asyncio_task(
            send_request_async, messages, self.req_socket
        )
        if future is None:
            logger.error("Future is None")
            return ""
        try:
            result = future.result()
            return result
        except Exception as e:
            logger.error(f"Error occurred when waiting for a response: {e}")
            return ""

    def disconnect(self):
        if self.device_info is None:
            logger.error(
                f"Device {self.device_name} is not "
                "connected and it cannot be disconnected"
            )
            return
        self.req_socket.disconnect(
            f"tcp://{self.device_info['ip']}:{self.device_info['port']}"
        )
        self.connected = False
        self.connected_node_name = None
        self.device_info = None
        self.device_id = None

    def print_log(self, log: str):
        logger.remote_log(f"{self.type} Log: {log}")

    def get_controller_data(self) -> InputData:
        raise NotImplementedError
