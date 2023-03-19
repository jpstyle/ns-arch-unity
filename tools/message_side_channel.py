"""
Simple Python-side implementation of custom Unity-Python side channels that
communicate string messages aside the main mlagents pipeline
"""
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    OutgoingMessage,
)
import uuid


class StringMsgChannel(SideChannel):
    def __init__(self, channel_uuid):
        super().__init__(uuid.UUID(channel_uuid))
        self.incoming_message_buffer = []

    def on_message_received(self, msg):
        self.incoming_message_buffer.append(msg.read_string())

    def send_string(self, data):
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
