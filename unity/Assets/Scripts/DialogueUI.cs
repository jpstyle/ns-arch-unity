using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class DialogueUI : MonoBehaviour
{
    // Serves as both user interface & inter-agent communication channel

    // UXML template for dialogue records
    public VisualTreeAsset dialogueRecordTemplate;

    // List of dialogue participants
    public List<DialogueAgent> dialogueParticipants;

    // UI element references
    VisualElement foldHandle;
    ListView dialogueHistoryView;
    VisualElement dialogueInput;
    DropdownField inputSpeakerChoiceField;
    TextField inputTextField;
    Button inputButton;

    // Dialogue record history buffer
    List<RecordData> dialogueRecords;

    // Demonstrative references for the current text input
    Dictionary<(int, int), string> demonstrativeReferences;

    void OnEnable()
    {
        // The UXML is already instantiated by the UIDocument component
        UIDocument uiDocument = GetComponent<UIDocument>();
        VisualElement root = uiDocument.rootVisualElement;

        dialogueRecords = new List<RecordData>();

        demonstrativeReferences = new Dictionary<(int, int), string>();

        // Store references to UI elements
        foldHandle = root.Q<VisualElement>("FoldHandle");
        dialogueHistoryView = root.Q<ListView>("DialogueHistory");
        dialogueInput = root.Q<VisualElement>("DialogueInput");
        inputSpeakerChoiceField = dialogueInput.Q<DropdownField>();
        inputTextField = dialogueInput.Q<TextField>();
        inputButton = dialogueInput.Q<Button>();

        // Register foldHandle callback
        foldHandle.RegisterCallback<ClickEvent>(FoldHandleToggle);

        // Register ListView callbacks & link data source
        dialogueHistoryView.makeItem = () =>
        {
            // Instantiate the UXML template for the entry
            TemplateContainer newRecordEntry = dialogueRecordTemplate.Instantiate();

            // Instantiate a controller for the entry data
            DialogueRecordController newRecordEntryLogic = new DialogueRecordController();

            // Assign the controller script to the visual element
            newRecordEntry.userData = newRecordEntryLogic;

            // Initialize the controller script
            newRecordEntryLogic.SetVisualElement(newRecordEntry);

            // Return the root of the instantiated visual tree
            return newRecordEntry;
        };
        dialogueHistoryView.bindItem = (entry, index) =>
        {
            (entry.userData as DialogueRecordController).SetRecordData(dialogueRecords[index]);
        };
        dialogueHistoryView.itemsSource = dialogueRecords;

        // Initialize speaker choice as empty list
        inputSpeakerChoiceField.choices = new List<string>();
        // Register input UI callbacks
        inputTextField.RegisterCallback<KeyDownEvent>(QueueUserUtteranceOnEnter);
        inputButton.RegisterCallback<ClickEvent>(QueueUserUtteranceOnClick);

        // Register participants to UI
        foreach (DialogueAgent agt in dialogueParticipants)
            RegisterParticipant(agt);

        // Add dialogue history header record
        CommitUtterance("System", "Start episode");
    }
    
    public void RegisterParticipant(DialogueAgent agent)
    {
        // Provide reference to this UI to participant
        agent.dialogueUI = this;

        // Add to speaker choice in dropdown
        inputSpeakerChoiceField.choices.Add(agent.dialogueParticipantID);

        // For development purpose, setting Teacher as default choice
        for (int i=0; i<inputSpeakerChoiceField.choices.Count; i++)
        {
            if (inputSpeakerChoiceField.choices[i].StartsWith("Teacher"))
            {
                inputSpeakerChoiceField.index = i;
                break;
            }
        }
        inputSpeakerChoiceField.index = inputSpeakerChoiceField.choices.IndexOf("Teacher");
    }

    public void FoldHandleToggle(ClickEvent evt)
    {
        Label handleLabel = foldHandle.Q<Label>();
        if (handleLabel.text.StartsWith("Fold"))
        {
            handleLabel.text = "Unfold ▼";
            dialogueHistoryView.style.display = DisplayStyle.None;
            dialogueInput.style.display = DisplayStyle.None;
        } else if (handleLabel.text.StartsWith("Unfold"))
        {
            handleLabel.text = "Fold ▲";
            dialogueHistoryView.style.display = DisplayStyle.Flex;
            dialogueInput.style.display = DisplayStyle.Flex;
        }
    }

    public void QueueUserUtteranceOnEnter(KeyDownEvent evt)
    {
        // Send focus to the input field
        inputTextField.Q("unity-text-input").Focus();

        string currentSpeaker = inputSpeakerChoiceField.choices[inputSpeakerChoiceField.index];
        string currentInput = inputTextField.text;

        if (evt.keyCode == KeyCode.Return && currentInput != "")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID == currentSpeaker)
                    agt.outgoingMsgBuffer.Enqueue(currentInput);
            }
            inputTextField.SetValueWithoutNotify("");
        }
    }

    public void QueueUserUtteranceOnClick(ClickEvent evt)
    {
        string currentSpeaker = inputSpeakerChoiceField.choices[inputSpeakerChoiceField.index];
        string currentInput = inputTextField.text;

        if (currentInput != "")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID == currentSpeaker)
                    agt.outgoingMsgBuffer.Enqueue(currentInput);
            }
            inputTextField.SetValueWithoutNotify("");
        }
    }

    public void CommitUtterance(string speaker, string inputString)
    {
        // Create a new record and add to list
        RecordData inputRecord = ScriptableObject.CreateInstance<RecordData>();
        inputRecord.speaker = speaker;
        inputRecord.utterance = inputString;
        dialogueRecords.Add(inputRecord);

        // Broadcast the record to all audience members
        if (speaker != "System")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID != speaker)
                    agt.incomingMsgBuffer.Enqueue(inputRecord);
            }
        }

        // Refresh view
        dialogueHistoryView.ScrollToItem(-1);
        dialogueHistoryView.Rebuild();
    }

    public void AddDemonstrativeWithReference()
    {
        
    }
}
