using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class MessageSideChannel : SideChannel
{
    // Associated agent in scene
    private readonly DialogueAgent _listeningAgent;

    public MessageSideChannel(string channelUuid, DialogueAgent agent)
    {
        ChannelId = new Guid(channelUuid);
        _listeningAgent = agent;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Read until end of message
        var utterance = msg.ReadString();

        // Retrieve any demonstrative references (map from substring indices to
        // segmentation mask image, having same dimension as scene image perceived
        // by sensor) until end of the current message is reached (signalled by -1)
        var demRefs = new Dictionary<(int, int), EntityRef>();
        while (true)
        {
            var intMessage = msg.ReadInt32();
        
            if (intMessage == -1) break;        // End of dem. refs list
            
            var start = intMessage;
            var end = msg.ReadInt32();
            var refByMask = msg.ReadBoolean();
            if (refByMask)
                demRefs[(start, end)] = new EntityRef(msg.ReadFloatList().ToArray());
            else
                demRefs[(start, end)] = new EntityRef(msg.ReadString());
        }

        // Put processed message data into incoming buffer
        _listeningAgent.outgoingMsgBuffer.Enqueue((utterance, demRefs));
    }

    public void SendMessageToBackend(
        string speaker, string utterance, Dictionary<(int, int), EntityRef> demRefs
    )
    {
        // Create OutgoingMessage instance (using for dispose at the end)
        using var msgOut = new OutgoingMessage();

        // Write speaker info & utterance content as string
        msgOut.WriteString(speaker);
        msgOut.WriteString(utterance);

        // (If any) Encode demonstrative references as two consecutive ints (marking
        // start & end of corresponding demonstrative pronoun substring) and either
        // float[] ((soft) segmentation mask) or string (direct reference by string
        // name of EnvEntity)
        foreach (var (range, demRef) in demRefs)
        {
            var (start, end) = range;
            msgOut.WriteInt32(start); msgOut.WriteInt32(end);
            switch (demRef.refType)
            {
                case EntityRefType.Mask:
                    msgOut.WriteBoolean(true);
                    msgOut.WriteFloatList(demRef.maskRef);
                    break;
                case EntityRefType.String:
                    msgOut.WriteBoolean(false);
                    msgOut.WriteString(demRef.stringRef);
                    break;
                default:
                    // Shouldn't reach here but anyways
                    throw new Exception("Invalid reference data type?");
            }
        }

        // Mark end of message segment
        msgOut.WriteInt32(-1);

        // Queue message to send
        QueueMessageToSend(msgOut);
    }
}
