using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

public class DialogueAgent : Agent
// Communication with dialogue UI and communication with Python backend may be
// decoupled later?
{
    // String identifier name for the dialogue participant agent
    public string dialogueParticipantID;

    // Commit utterances through this dialogue UI
    [HideInInspector]
    public DialogueUI dialogueUI;

    // Message communication buffer queues
    public readonly Queue<RecordData> IncomingMsgBuffer = new Queue<RecordData>();
    public readonly Queue<string> OutgoingMsgBuffer = new Queue<string>();

    // Communication side channel to Python backend for requesting decisions
    protected string ChannelUuid;
    protected StringMsgSideChannel BackendMsgChannel;

    // Only for use by non-Heuristics agents (i.e. connected to Python backend);
    // boolean flag for checking whether the agent has some unresolved goal to
    // achieve and thus needs RequestDecision() call
    // protected bool HasGoalToResolve = false;

    // For controlling minimal update interval, to allow visual inspection during runs
    float _time;
    readonly float _tInterval = 0.1f;

    public void Start()
    {
        _time = Time.time;
    }

    public void Update()
    {
        if (GetComponent<BehaviorParameters>().BehaviorType == BehaviorType.HeuristicOnly)
        {
            // Always empty incoming message buffer and call RequestDecision
            IncomingMsgBuffer.Clear();
            RequestDecision();
        }
        else
        {
            if (Time.time > _time)
            {
                _time += _tInterval;

                // Trying to consult backend for requesting decision only when needed, in order
                // to minimize communication of visual observation data
                if (IncomingMsgBuffer.Count > 0 || OutgoingMsgBuffer.Count > 0) //|| HasGoalToResolve)
                {
                    while (IncomingMsgBuffer.Count > 0)
                    {
                        RecordData incomingMessage = IncomingMsgBuffer.Dequeue();
                        BackendMsgChannel.SendMessageToBackend(incomingMessage.speaker);
                        BackendMsgChannel.SendMessageToBackend(incomingMessage.utterance);
                    }

                    RequestDecision();
                }
            }
        }
    }

    protected void Utter()
    {
        while (OutgoingMsgBuffer.Count > 0) {
            string outgoingMessage = OutgoingMsgBuffer.Dequeue();
            dialogueUI.CommitUtterance(dialogueParticipantID, outgoingMessage);
        }
    }
}
