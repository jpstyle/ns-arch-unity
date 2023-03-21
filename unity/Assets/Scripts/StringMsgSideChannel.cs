using System;
using Unity.MLAgents.SideChannels;

public class StringMsgSideChannel : SideChannel
{
    // Associated agent in scene
    readonly DialogueAgent _listeningAgent;

    public StringMsgSideChannel(string channelUuid, DialogueAgent agent)
    {
        ChannelId = new Guid(channelUuid);
        _listeningAgent = agent;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        _listeningAgent.OutgoingMsgBuffer.Enqueue(msg.ReadString());
    }

    public void SendMessageToBackend(string msgString)
    {
        OutgoingMessage msgOut = new OutgoingMessage();
        msgOut.WriteString(msgString);
        QueueMessageToSend(msgOut);
        msgOut.Dispose();
    }
}
