using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;

public class TeacherAgent : DialogueAgent
{
    protected override void Awake()
    {
        // Register Python-Agent string communication side channel
        // (Note: This works because we will have only one instance of the agent
        // in the scene ever, but ideally we would want 1 channel per instance,
        // with UUIDs generated on the fly each time an instance is created...)
        channelUuid = "da85d4e0-1b60-4c8a-877d-03af30c446f2";
        backendMsgChannel = new MessageSideChannel(channelUuid, this);
        SideChannelManager.RegisterSideChannel(backendMsgChannel);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (actionBuffers.DiscreteActions[0] == 1)
        {
            // 'Utter' action
            Utter();
        }
    }

    public override void Heuristic(in ActionBuffers actionBuffers)
    {
        // 'Utter' any outgoing messages
        if (outgoingMsgBuffer.Count > 0)
        {
            var discreteActionBuffers = actionBuffers.DiscreteActions;
            discreteActionBuffers[0] = 1;      // 'Utter'
        }
    }
}
