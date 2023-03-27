using System;
using System.Collections.Generic;
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
        // Speaker info & utterance content as string
        var utterance = msg.ReadString();

        // Retrieve any demonstrative references (map from substring indices to
        // relative box coordinates on visual observation image) until end of the
        // current message is reached (signalled by -1)
        var demRefs = new Dictionary<(int, int), Rect>();
        while (true)
        {
            var intMessage = msg.ReadInt32();
            
            if (intMessage == -1)
                break;
            else
            {
                var start = intMessage;
                var end = msg.ReadInt32();
                var boxCoordinates = msg.ReadFloatList();
                demRefs[(start, end)] = new Rect(
                    boxCoordinates[0], boxCoordinates[1],
                    boxCoordinates[2], boxCoordinates[3]
                );
            }
        }
        
        // Put processed message data into incoming buffer
        _listeningAgent.outgoingMsgBuffer.Enqueue((utterance, demRefs));
    }

    public void SendMessageToBackend(
        string speaker, string utterance, Dictionary<(int, int), Rect> demRefs
    )
    {
        // Create OutgoingMessage instance (using for dispose at the end)
        using var msgOut = new OutgoingMessage();
        
        // Write speaker info & utterance content as string
        msgOut.WriteString(speaker);
        msgOut.WriteString(utterance);
        
        // (If any) Encode demonstrative references as two consecutive ints (marking
        // start & end of corresponding demonstrative pronoun substring) and float[4]
        // (marking box coordinates relative to visual obs)
        foreach (var (range, rect) in demRefs)
        {
            var (start, end) = range;
            msgOut.WriteInt32(start); msgOut.WriteInt32(end);
            msgOut.WriteFloatList(new[] {rect.x, rect.y, rect.width, rect.height});
        }

        // Mark end of message
        msgOut.WriteInt32(-1);
        
        // Queue message to send
        QueueMessageToSend(msgOut);
    }
}
