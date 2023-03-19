using System;
using UnityEngine;
using Unity.MLAgents.SideChannels;

public class StringMsgSideChannel : SideChannel
{
    // Associated agent in scene
    DialogueAgent listeningAgent;

    public StringMsgSideChannel(string channelUuid, DialogueAgent agent)
    {
        ChannelId = new Guid(channelUuid);
        listeningAgent = agent;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        listeningAgent.outgoingMsgBuffer.Enqueue(msg.ReadString());
    }

    public void SendMessageToBackend(string msgString)
    {
        using (OutgoingMessage msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(msgString);
            QueueMessageToSend(msgOut);
            msgOut.Dispose();
        }
    }
}
