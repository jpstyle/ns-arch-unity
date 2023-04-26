using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
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
    public readonly Queue<RecordData> incomingMsgBuffer = new();
    public readonly Queue<(string, Dictionary<(int, int), EntityRef>)> outgoingMsgBuffer = new();

    // Communication side channel to Python backend for requesting decisions
    protected string channelUuid;
    protected MessageSideChannel backendMsgChannel;
    
    // Camera sensor component
    private CameraSensorComponent _cameraSensor;

    // Behavior type as MLAgent
    private BehaviorType _behaviorType;

    // // Only for use by non-Heuristics agents (i.e. connected to Python backend);
    // // boolean flag for checking whether the agent has some unresolved goal to
    // // achieve and thus needs RequestDecision() call
    // protected bool HasGoalToResolve = false;

    // For controlling minimal update interval, to allow visual inspection during runs
    private float _nextTimeToAct;
    private const float TimeInterval = 2f;

    public void Start()
    {
        _cameraSensor = GetComponent<CameraSensorComponent>();
        _behaviorType = GetComponent<BehaviorParameters>().BehaviorType;
        _nextTimeToAct = Time.time;
    }

    public override void OnEpisodeBegin()
    {
        Utter();
    }

    public void Update()
    {
        if (Time.time < _nextTimeToAct) return;
        
        _nextTimeToAct += TimeInterval;
        if (_behaviorType == BehaviorType.HeuristicOnly)
        {
            // Always empty incoming message buffer and call RequestDecision
            incomingMsgBuffer.Clear();
            RequestDecision();
        }
        else
        {
            // Trying to consult backend for requesting decision only when needed, in order
            // to minimize communication of visual observation data
            if (incomingMsgBuffer.Count == 0) return;

            // Unprocessed incoming messages exist; process and consult backend
            var speakers = new List<string>();
            var utterances = new List<string>();
            var demRefs = new List<Dictionary<(int, int), EntityRef>>();

            while (incomingMsgBuffer.Count > 0)
            {
                // Fetch single message record from queue
                var incomingMessage = incomingMsgBuffer.Dequeue();
                
                // (If any) Translate EnvEntity reference by UID to box coordinates w.r.t.
                // this agent's camera sensor
                var demRefsForUtt = new Dictionary<(int, int), EntityRef>();
                foreach (var (range, entUid) in incomingMessage.demonstrativeReferences)
                {
                    // Retrieve referenced EnvEntity and fetch absolute box coordinates w.r.t.
                    // this agent's camera's target display screen
                    var refEnt = EnvEntity.FindByUid(entUid);
                    var screenAbsRect = refEnt.boxes[_cameraSensor.Camera.targetDisplay];
                    demRefsForUtt[range] = new EntityRef(ScreenAbsRectToSensorRelRect(screenAbsRect));
                }

                speakers.Add(incomingMessage.speaker);
                utterances.Add(incomingMessage.utterance);
                demRefs.Add(demRefsForUtt);
            }

            // Now send message via side channel and wait for decision
            backendMsgChannel.SendMessageToBackend(speakers, utterances, demRefs);
            RequestDecision();
        }
    }

    protected void Utter()
    {
        while (outgoingMsgBuffer.Count > 0) {
            var (utterance, demRefs) = outgoingMsgBuffer.Dequeue();

            if (demRefs is not null && demRefs.Count > 0)
            {
                // Need to resolve demonstrative reference boxes to corresponding EnvEntity (uid)
                var demRefsResolved = new Dictionary<(int, int), string>();
                var targetDisplay = _cameraSensor.Camera.targetDisplay;

                foreach (var (range, demRef) in demRefs)
                {
                    switch (demRef.refType)
                    {
                        case EntityRefType.BBox:
                            var screenAbsRect = SensorRelRectToScreenAbsRect(demRef.bboxRef);
                            demRefsResolved[range] = EnvEntity.FindByBox(screenAbsRect, targetDisplay).uid;
                            break;
                        case EntityRefType.String:
                            demRefsResolved[range] = EnvEntity.FindByObjectName(demRef.stringRef).uid;
                            break;
                        default:
                            // Shouldn't reach here but anyways
                            throw new Exception("Invalid reference data type?");
                    }
                }

                dialogueUI.CommitUtterance(dialogueParticipantID, utterance, demRefsResolved);
            }
            else
                // No demonstrative references to process and resolve
                dialogueUI.CommitUtterance(dialogueParticipantID, utterance);
        }
    }

    // To be overridden by children classes, do nothing
    protected virtual void InitiateInteraction() {}

    private Rect ScreenAbsRectToSensorRelRect(Rect screenAbsRect)
    {
        var targetDisplay = _cameraSensor.Camera.targetDisplay;
        var screenWidth = Display.displays[targetDisplay].systemWidth;
        var screenHeight = Display.displays[targetDisplay].systemHeight;

        // To screen-relative coordinates
        var screenRelRectX = screenAbsRect.x / screenWidth;
        var screenRelRectY = screenAbsRect.y / screenHeight;
        var screenRelRectWidth = screenAbsRect.width / screenWidth;
        var screenRelRectHeight = screenAbsRect.height / screenHeight;
                    
        // To CameraSensor-relative coordinates; transform x-coordinates according to
        // ratio of aspect ratios, while leaving y-coordinates untouched
        var screenAspectRatio = (float)screenWidth / screenHeight;
        var sensorAspectRatio = (float)_cameraSensor.Width / _cameraSensor.Height;
        var arRatio = screenAspectRatio / sensorAspectRatio;
        var sensorRelRectX = (screenRelRectX - 0.5f) * arRatio + 0.5f;
        var sensorRelRectWidth = screenRelRectWidth * arRatio;

        return new Rect(sensorRelRectX, screenRelRectY, sensorRelRectWidth, screenRelRectHeight);
    }
    
    private Rect SensorRelRectToScreenAbsRect(Rect sensorRelRect)
    {
        var targetDisplay = _cameraSensor.Camera.targetDisplay;
        var screenWidth = Display.displays[targetDisplay].systemWidth;
        var screenHeight = Display.displays[targetDisplay].systemHeight;

        // To screen-relative coordinates; transform x-coordinates according to
        // ratio of aspect ratios, while leaving y-coordinates untouched
        var screenAspectRatio = (float)screenWidth / screenHeight;
        var sensorAspectRatio = (float)_cameraSensor.Width / _cameraSensor.Height;
        var arRatio = screenAspectRatio / sensorAspectRatio;
        var screenRelRectX = (sensorRelRect.x - 0.5f) / arRatio + 0.5f;
        var screenRelRectY = sensorRelRect.y;
        var screenRelRectWidth = sensorRelRect.width / arRatio;
        var screenRelRectHeight = sensorRelRect.height;

        // To screen-absolute coordinates
        var screenAbsRectX = screenRelRectX * screenWidth;
        var screenAbsRectY = screenRelRectY * screenHeight;
        var screenAbsRectWidth = screenRelRectWidth * screenWidth;
        var screenAbsRectHeight = screenRelRectHeight * screenHeight;

        return new Rect(screenAbsRectX, screenAbsRectY, screenAbsRectWidth, screenAbsRectHeight);
    }
}
