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
        # segmentation masks on visual observation image) until end of the
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
                ref_by_mask = msg.read_bool()
                if ref_by_mask:
                    # Reference by segmentation mask (list of floats in 0~1)
                    dem_refs[(start, end)] = self.rle_decode(msg.read_float32_list())
                else:
                    # Reference by string name
                    dem_refs[(start, end)] = msg.read_string()

        # Put processed message data into incoming buffer
        self.incoming_message_buffer.append((speaker, utterance, dem_refs))

    def send_string(self, speaker, utterance, dem_refs):
        assert isinstance(utterance, str) and isinstance(dem_refs, dict)

        msg = OutgoingMessage()

        # Utterance content as string
        msg.write_string(speaker)
        msg.write_string(utterance)

        # Write any demonstrative references
        for (start, end), ref in dem_refs.items():
            msg.write_int32(start); msg.write_int32(end)
            if isinstance(ref, list):
                # Reference by segmentation mask (list of floats in 0~1)
                assert all(x>=0 and x<=1 for x in ref)
                msg.write_bool(True)
                msg.write_float32_list(self.rle_encode(ref))
            else:
                # Reference by string name (of Unity GameObject)
                assert isinstance(ref, str)
                msg.write_bool(False)
                msg.write_string(ref)

        # Mark end of message segment
        msg.write_int32(-1)

        # Queue message to send
        super().queue_message_to_send(msg)

    @staticmethod
    def rle_encode(raw_mask):
        # Encode raw binary mask into RLE format for message compression
        rle = []

        zero_flag = True
        run = 0
        for f in raw_mask:
            # Increment run length if value matches with current flag; push current
            # run length value to return array and flip sign
            if zero_flag:
                if f == 0: run += 1
                else:
                    rle.append(run)
                    zero_flag = False
                    run = 1
            else:
                if f > 0: run += 1
                else:
                    rle.append(run)
                    zero_flag = True
                    run = 1
        if run > 0: rle.append(run)     # Flush last entry

        return rle

    @staticmethod
    def rle_decode(rle_mask):
        # Decode RLE to recover raw binary mask
        total_length = int(sum(rle_mask))
        raw = [None] * total_length

        zero_flag = True
        cumulative = 0
        for f in rle_mask:
            # Get integer run length and update values
            run_length = int(f)
            for i in range(cumulative, cumulative+run_length):
                raw[i] = 0 if zero_flag else 1
            
            # Flip sign and update cumulative index
            zero_flag = not zero_flag
            cumulative += run_length

        return raw
