using UnityEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;

public class StudentAgent : DialogueAgent
{
    // For visualizing handheld objects
    public Transform leftHand;
    public Transform rightHand;

    protected override void Awake()
    {
        // Register Python-Agent string communication side channel
        // (Note: This works because we will have only one instance of the agent
        // in the scene ever, but ideally we would want 1 channel per instance,
        // with UUIDs generated on the fly each time an instance is created...)
        ChannelUuid = "a1a6b269-0dd3-442c-99c6-9c735ebe43e1";
        BackendMsgChannel = new StringMsgSideChannel(ChannelUuid, this);
        SideChannelManager.RegisterSideChannel(BackendMsgChannel);
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
        if (OutgoingMsgBuffer.Count > 0)
        {
            ActionSegment<int> discreteActionBuffers = actionBuffers.DiscreteActions;
            discreteActionBuffers[0] = 1;      // 'Utter'
        }
    }
}
