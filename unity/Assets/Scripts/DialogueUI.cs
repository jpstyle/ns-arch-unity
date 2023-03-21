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
    VisualElement _foldHandle;
    ListView _dialogueHistoryView;
    VisualElement _dialogueInput;
    DropdownField _inputSpeakerChoiceField;
    TextField _inputTextField;
    Button _inputButton;

    // Dialogue record history buffer
    List<RecordData> _dialogueRecords;

    // Demonstrative references for the current text input
    Dictionary<(int, int), string> _demonstrativeReferences;

    void OnEnable()
    {
        // The UXML is already instantiated by the UIDocument component
        UIDocument uiDocument = GetComponent<UIDocument>();
        VisualElement root = uiDocument.rootVisualElement;

        _dialogueRecords = new List<RecordData>();

        _demonstrativeReferences = new Dictionary<(int, int), string>();

        // Store references to UI elements
        _foldHandle = root.Q<VisualElement>("FoldHandle");
        _dialogueHistoryView = root.Q<ListView>("DialogueHistory");
        _dialogueInput = root.Q<VisualElement>("DialogueInput");
        _inputSpeakerChoiceField = _dialogueInput.Q<DropdownField>();
        _inputTextField = _dialogueInput.Q<TextField>();
        _inputButton = _dialogueInput.Q<Button>();

        // Register foldHandle callback
        _foldHandle.RegisterCallback<ClickEvent>(FoldHandleToggle);

        // Register ListView callbacks & link data source
        _dialogueHistoryView.makeItem = () =>
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
        _dialogueHistoryView.bindItem = (entry, index) =>
        {
            (entry.userData as DialogueRecordController)?.SetRecordData(_dialogueRecords[index]);
        };
        _dialogueHistoryView.itemsSource = _dialogueRecords;

        // Initialize speaker choice as empty list
        _inputSpeakerChoiceField.choices = new List<string>();
        // Register input UI callbacks
        _inputTextField.RegisterCallback<KeyDownEvent>(QueueUserUtteranceOnEnter);
        _inputButton.RegisterCallback<ClickEvent>(QueueUserUtteranceOnClick);

        // Register participants to UI
        foreach (DialogueAgent agt in dialogueParticipants)
            RegisterParticipant(agt);

        // Add dialogue history header record
        CommitUtterance("System", "Start episode");
    }

    private void RegisterParticipant(DialogueAgent agent)
    {
        // Provide reference to this UI to participant
        agent.dialogueUI = this;

        // Add to speaker choice in dropdown
        _inputSpeakerChoiceField.choices.Add(agent.dialogueParticipantID);

        // For development purpose, setting Teacher as default choice
        for (int i=0; i<_inputSpeakerChoiceField.choices.Count; i++)
        {
            if (_inputSpeakerChoiceField.choices[i].StartsWith("Teacher"))
            {
                _inputSpeakerChoiceField.index = i;
                break;
            }
        }
        _inputSpeakerChoiceField.index = _inputSpeakerChoiceField.choices.IndexOf("Teacher");
    }

    private void FoldHandleToggle(ClickEvent evt)
    {
        Label handleLabel = _foldHandle.Q<Label>();
        if (handleLabel.text.StartsWith("Fold"))
        {
            handleLabel.text = "Unfold ▼";
            _dialogueHistoryView.style.display = DisplayStyle.None;
            _dialogueInput.style.display = DisplayStyle.None;
        } else if (handleLabel.text.StartsWith("Unfold"))
        {
            handleLabel.text = "Fold ▲";
            _dialogueHistoryView.style.display = DisplayStyle.Flex;
            _dialogueInput.style.display = DisplayStyle.Flex;
        }
    }

    private void QueueUserUtteranceOnEnter(KeyDownEvent evt)
    {
        // Send focus to the input field
        _inputTextField.Q("unity-text-input").Focus();

        string currentSpeaker = _inputSpeakerChoiceField.choices[_inputSpeakerChoiceField.index];
        string currentInput = _inputTextField.text;

        if (evt.keyCode == KeyCode.Return && currentInput != "")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID == currentSpeaker)
                    agt.OutgoingMsgBuffer.Enqueue(currentInput);
            }
            _inputTextField.SetValueWithoutNotify("");
        }
    }

    private void QueueUserUtteranceOnClick(ClickEvent evt)
    {
        string currentSpeaker = _inputSpeakerChoiceField.choices[_inputSpeakerChoiceField.index];
        string currentInput = _inputTextField.text;

        if (currentInput != "")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID == currentSpeaker)
                    agt.OutgoingMsgBuffer.Enqueue(currentInput);
            }
            _inputTextField.SetValueWithoutNotify("");
        }
    }

    public void CommitUtterance(string speaker, string inputString)
    {
        // Create a new record and add to list
        RecordData inputRecord = ScriptableObject.CreateInstance<RecordData>();
        inputRecord.speaker = speaker;
        inputRecord.utterance = inputString;
        _dialogueRecords.Add(inputRecord);

        // Broadcast the record to all audience members
        if (speaker != "System")
        {
            foreach (DialogueAgent agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID != speaker)
                    agt.IncomingMsgBuffer.Enqueue(inputRecord);
            }
        }

        // Refresh view
        _dialogueHistoryView.ScrollToItem(-1);
        _dialogueHistoryView.Rebuild();
    }

    public void AddDemonstrativeWithReference()
    {
        
    }
}
