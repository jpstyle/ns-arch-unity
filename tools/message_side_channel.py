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
        # Read until end of message

        # Speaker info & utterance content as string
        speaker = msg.read_string()
        utterance = msg.read_string()

        # Retrieve any demonstrative references (map from substring indices to
        # relative box coordinates on visual observation image) until end of the
        # current message is reached (signalled by -1)
        dem_refs = {}
        while True:
            int_message = msg.read_int32()

            if int_message == -1:
                # End of message
                break
            else:
                # Read integer is substring start index; process the remainder
                start = int_message
                end = msg.read_int32()
                ref_by_bbox = msg.read_bool()
                if ref_by_bbox:
                    # Reference by bounding box coordinates (list of ffloats)
                    dem_refs[(start, end)] = msg.read_float32_list()
                else:
                    # Reference by string name
                    dem_refs[(start, end)] = msg.read_string()

        # Put processed message data into incoming buffer
        self.incoming_message_buffer.append((speaker, utterance, dem_refs))

    def send_string(self, utterance, dem_refs):
        assert isinstance(utterance, str) and isinstance(dem_refs, dict)

        msg = OutgoingMessage()

        # Utterance content as string
        msg.write_string(utterance)

        # Write any demonstrative references
        for (start, end), ref in dem_refs.items():
            msg.write_int32(start); msg.write_int32(end)
            if isinstance(ref, list):
                # Reference by bounding box coordinates (list of ffloats)
                assert len(ref)==4 and all(isinstance(x, float) for x in ref)
                msg.write_bool(True)
                msg.write_float32_list(ref)
            else:
                # Reference by string name (of Unity GameObject)
                assert isinstance(ref, str)
                msg.write_bool(False)
                msg.write_string(ref)

        # Mark end of message segment
        msg.write_int32(-1)

        # Queue message to send
        super().queue_message_to_send(msg)
