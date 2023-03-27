using System;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class StringMsgSideChannel : SideChannel
{
    // Associated agent in scene
    private readonly DialogueAgent _listeningAgent;

    public StringMsgSideChannel(string channelUuid, DialogueAgent agent)
    {
        ChannelId = new Guid(channelUuid);
        _listeningAgent = agent;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        _listeningAgent.outgoingMsgBuffer.Enqueue(msg.ReadString());
    }

    public void SendMessageToBackend(string msgString)
    {
        using var msgOut = new OutgoingMessage();
        msgOut.WriteString(msgString);
        QueueMessageToSend(msgOut);
    }
}
