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
        channelUuid = "a1a6b269-0dd3-442c-99c6-9c735ebe43e1";
        backendMsgChannel = new MessageSideChannel(channelUuid, this);
        SideChannelManager.RegisterSideChannel(backendMsgChannel);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (actionBuffers.DiscreteActions[0] == 1)
        {
            // 'Utter' action
            StartCoroutine(Utter());
        }
    }

    public override void Heuristic(in ActionBuffers actionBuffers)
    {
        // Update annotation info whenever needed
        if (!EnvEntity.annotationStorage.annotationsUpToDate)
            StartCoroutine(CaptureAnnotations());

        // 'Utter' any outgoing messages
        if (outgoingMsgBuffer.Count > 0)
        {
            var discreteActionBuffers = actionBuffers.DiscreteActions;
            discreteActionBuffers[0] = 1;      // 'Utter'
        }
    }
}
