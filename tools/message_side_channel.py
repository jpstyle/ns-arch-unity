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
        # Speaker info & utterance content as string
        speaker = msg.read_string()
        utterance = msg.read_string()

        # Retrieve any demonstrative references (map from substring indices to
        # relative box coordinates on visual observation image) until end of the
        # current message is reached (signalled by -1)
        demRefs = {}
        while True:
            int_message = msg.read_int32()

            if int_message == -1:
                # End of message
                break
            else:
                # Read integer is substring start index; process the remainder
                start = int_message
                end = msg.read_int32()
                box_coordinates = msg.read_float32_list()
                demRefs[(start, end)] = box_coordinates

        # Put processed message data into incoming buffer
        self.incoming_message_buffer.append((speaker, utterance, demRefs))

    def send_string(self, utterance, demRefs):
        msg = OutgoingMessage()

        # Utterance content as string
        msg.write_string(utterance)

        # Write any demonstrative references
        for (start, end), bbox in demRefs.items():
            msg.write_int32(start); msg.write_int32(end)
            msg.write_float32_list(bbox)

        # Mark end of message
        msg.write_int32(-1)

        # Queue message to send
        super().queue_message_to_send(msg)
