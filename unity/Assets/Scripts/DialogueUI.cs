using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using spkl.Diffs;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.UIElements.Experimental;

public class DialogueUI : MonoBehaviour
{
    // Serves as both user interface & inter-agent communication channel

    // UXML template for dialogue records
    public VisualTreeAsset dialogueRecordTemplate;

    // List of dialogue participants
    public List<DialogueAgent> dialogueParticipants;

    // UI element references
    private VisualElement _foldHandle;
    private ListView _dialogueHistoryView;
    private VisualElement _dialogueInput;
    private DropdownField _inputSpeakerChoiceField;
    private TextField _inputTextField;
    private Button _inputButton;
    private Label _inputDemRefView;

    // Dialogue record history buffer
    private List<RecordData> _dialogueRecords;

    // Demonstrative references for the current text input
    private Dictionary<(int, int), string> _demonstrativeReferences;

    private void OnEnable()
    {
        // The UXML is already instantiated by the UIDocument component
        var uiDocument = GetComponent<UIDocument>();
        var root = uiDocument.rootVisualElement;

        _dialogueRecords = new List<RecordData>();

        _demonstrativeReferences = new Dictionary<(int, int), string>();

        // Store references to UI elements
        _foldHandle = root.Q<VisualElement>("FoldHandle");
        _dialogueHistoryView = root.Q<ListView>("DialogueHistory");
        _dialogueInput = root.Q<VisualElement>("DialogueInput");
        _inputSpeakerChoiceField = _dialogueInput.Q<DropdownField>();
        _inputTextField = _dialogueInput.Q<TextField>();
        _inputButton = _dialogueInput.Q<Button>();
        _inputDemRefView = root.Q<Label>("InputDemRefs");

        // Register foldHandle callback
        _foldHandle.RegisterCallback<ClickEvent>(FoldHandleToggle);

        // Register ListView callbacks & link data source
        _dialogueHistoryView.makeItem = () =>
        {
            // Instantiate the UXML template for the entry
            var newRecordEntry = dialogueRecordTemplate.Instantiate();

            // Instantiate a controller for the entry data
            var newRecordEntryLogic = new DialogueRecordController();

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
        _inputTextField.RegisterCallback<ChangeEvent<string>>(MonitorDemonstrativeReferences);
        _inputButton.RegisterCallback<ClickEvent>(QueueUserUtteranceOnClick);
        
        // Register demRef view PointerLinkTagEvent callbacks
        _inputDemRefView.RegisterCallback<PointerOverLinkTagEvent>(HighlightReference);
        _inputDemRefView.RegisterCallback<PointerOutLinkTagEvent>(RemoveReferenceHighlight);

        // Register participants to UI
        foreach (var agt in dialogueParticipants)
            RegisterParticipant(agt);
    }

    private void RegisterParticipant(DialogueAgent agent)
    {
        // Provide reference to this UI to participant
        agent.dialogueUI = this;

        // Add to speaker choice in dropdown
        _inputSpeakerChoiceField.choices.Add(agent.dialogueParticipantID);

        // For development purpose, setting Teacher as default choice
        for (var i=0; i<_inputSpeakerChoiceField.choices.Count; i++)
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
        var handleLabel = _foldHandle.Q<Label>();
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
        
        var currentInput = _inputTextField.text;
        if (evt.keyCode != KeyCode.Return || currentInput == "") return;

        var currentSpeaker = _inputSpeakerChoiceField.choices[_inputSpeakerChoiceField.index];
        
        foreach (var agt in dialogueParticipants)
        {
            if (agt.dialogueParticipantID == currentSpeaker)
                agt.outgoingMsgBuffer.Enqueue((agt.dialogueParticipantID, currentInput, null));
        }
        
        // Clear input field and demRefs view
        _inputTextField.SetValueWithoutNotify("");
        _inputDemRefView.style.display = DisplayStyle.None;
    }

    private void QueueUserUtteranceOnClick(ClickEvent evt)
    {
        var currentInput = _inputTextField.text;
        if (currentInput == "") return;

        var currentSpeaker = _inputSpeakerChoiceField.choices[_inputSpeakerChoiceField.index];
        
        foreach (var agt in dialogueParticipants)
        {
            if (agt.dialogueParticipantID == currentSpeaker)
                agt.outgoingMsgBuffer.Enqueue((agt.dialogueParticipantID, currentInput, null));
        }
        
        // Clear input field and demRefs view
        _inputTextField.SetValueWithoutNotify("");
        _inputDemRefView.style.display = DisplayStyle.None;
    }
    
    private void MonitorDemonstrativeReferences(ChangeEvent<string> evt)
    {
        // Check if the value change 'compromises' a demonstrative reference; if some string
        // is deleted or inserted in the range of an existing demonstrative pronoun, consider
        // the reference as cancelled and remove it from current dictionary

        // Below is not needed if dictionary is empty
        if (_demonstrativeReferences.Count == 0) return;

        // Estimate the input edit with Myers diff algorithm (evt doesn't directly provide
        // what was the edit operation that led to the change from previousValue->newValue)
        var prevVal = evt.previousValue.ToCharArray();
        var newVal = evt.newValue.ToCharArray();
        var diff = new MyersDiff<char>(prevVal, newVal);

        // Keep track of compromised demRefs, so that they can be deleted
        var toDelete = new List<(int, int)>();
        
        // Keep track of demRefs whose indices must be shifted due to edits to their left
        var toLShift = new Dictionary<(int, int), int>();

        foreach (var instr in diff.GetEditScript())
        {
            var (indA, indB, cntA, cntB) = instr;

            foreach (var range in _demonstrativeReferences.Keys)
            {
                var (start, end) = range;
                
                // DemRefs to delete
                if (cntA > 0 && indA < end && indA+cntA > start)
                    toDelete.Add(range);   // Compromise by deletion
                if (cntB > 0 && start < indB && indA < end)
                    toDelete.Add(range);   // Compromise by insertion
                
                // DemRefs to shift indices
                if (cntA > 0 && indA < start && !toDelete.Contains(range))
                {
                    // Shift to left by cntA
                    if (toLShift.ContainsKey(range))
                        toLShift[range] += -cntA;
                    else
                        toLShift.Add(range, -cntA);
                        
                }
                if (cntB > 0 && indB <= start && !toDelete.Contains(range))
                {
                    // Shift to left by cntB
                    if (toLShift.ContainsKey(range))
                        toLShift[range] += cntB;
                    else
                        toLShift.Add(range, cntB);
                }
            }
        }

        // If any demRef is to be removed or left-shifted, create and assign a new dictionary
        // from existing one to reflect such changes
        if (toDelete.Count > 0 || toLShift.Count > 0)
        {
            var newDict = new Dictionary<(int, int), string>();
            foreach (var (range, refEntId) in _demonstrativeReferences)
            {
                if (toDelete.Contains(range)) continue;     // Omit in new dictionary
                if (toLShift.TryGetValue(range, out var offset))
                {
                    var (start, end) = range;
                    newDict.Add((start + offset, end + offset), refEntId);
                }
                else
                {
                    newDict.Add(range, refEntId);
                }
            }

            _demonstrativeReferences = newDict;
        }

        // Update view
        UpdateDemRefView();
    }
    
    private void HighlightReference(PointerOverLinkTagEvent evt)
    {
        var ent = EnvEntity.FindByUid(evt.linkID);
        
        var pointerUIs = FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.HighlightEnt(ent);
    }

    private void RemoveReferenceHighlight(PointerOutLinkTagEvent evt)
    {
        var pointerUIs = FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.RemoveHighlight();
    }

    private void UpdateDemRefView()
    {
        if (_demonstrativeReferences.Count > 0)
        {
            var demRefViews = new List<string>();
            foreach (var (range, refEntId) in _demonstrativeReferences)
            {
                // Data to visualize
                var (start, end) = range;
                var demPron = _inputTextField.text[start..end];
                var refEnt = EnvEntity.FindByUid(refEntId);

                // Label text pieces with rich tags
                var rangeString = $"<color=#808080>({start}-{end})</color>";
                var referenceLinkString = $"<color=#FF9600><link=\"{refEnt.uid}\">[{refEnt.gameObject.name}]</link></color>";
                
                demRefViews.Add($"'{demPron}' {rangeString}: {referenceLinkString}");
            }

            _inputDemRefView.style.display = DisplayStyle.Flex;
            _inputDemRefView.text = string.Join(", ", demRefViews);
        }
        else
        {
            _inputDemRefView.style.display = DisplayStyle.None;
            _inputDemRefView.text = "";
        }
    }

    public void AddDemonstrativeWithReference(string uid)
    {
        var selectionStart = Math.Min(_inputTextField.cursorIndex, _inputTextField.selectIndex);
        var selectionEnd = Math.Max(_inputTextField.cursorIndex, _inputTextField.selectIndex);

        // By default, using 'this' as the demonstrative pronoun; I guess there's
        // not much point in varying this, at least for now...
        var demPron = selectionStart == 0 ? "This" : "this";

        // Update input text field value & add to current map of demonstrative references
        var newInputPrefix = _inputTextField.text[..selectionStart];
        var newInputSuffix = _inputTextField.text[selectionEnd..];
        if (newInputPrefix.Length > 0 && !newInputPrefix.EndsWith(' '))
        {
            demPron = " " + demPron;
            selectionStart += 1;
        }
        if (newInputSuffix.Length > 0 && !newInputSuffix.StartsWith(' '))
        {
            demPron += " ";
        }
        _inputTextField.SetValueWithoutNotify(newInputPrefix + demPron + newInputSuffix);
        _inputTextField.cursorIndex = selectionStart + 4;
        _inputTextField.selectIndex = selectionStart + 4;
        _demonstrativeReferences[(selectionStart, selectionStart + 4)] = uid;
        
        // Update view
        UpdateDemRefView();
    }

    public void CommitUtterance(
        string speaker, string inputString, Dictionary<(int, int), string> optionalDemRefs = null
    )
    {
        // Create a new record and add to list, renew demRefs dictionary
        var inputRecord = ScriptableObject.CreateInstance<RecordData>();
        inputRecord.speaker = speaker;
        inputRecord.utterance = inputString;
        if (optionalDemRefs is null)
        {
            // Use current demRefs in dialogue UI
            inputRecord.demonstrativeReferences =
                new ReadOnlyDictionary<(int, int), string>(_demonstrativeReferences);
            _demonstrativeReferences = new Dictionary<(int, int), string>();
        }
        else
        {
            // Use provided demRefs, without refreshing current one in dialogue UI
            inputRecord.demonstrativeReferences =
                new ReadOnlyDictionary<(int, int), string>(optionalDemRefs);
        }
        _dialogueRecords.Add(inputRecord);

        // Broadcast the record to all audience members
        if (speaker != "System")
        {
            foreach (var agt in dialogueParticipants)
            {
                if (agt.dialogueParticipantID != speaker)
                    agt.incomingMsgBuffer.Enqueue(inputRecord);
            }
        }
        
        // Refresh view
        _dialogueHistoryView.Rebuild();
        _dialogueHistoryView.ScrollToItemById(-1);
    }

    public void ClearHistory()
    {
        _dialogueRecords.Clear();
        UpdateDemRefView();
    }
}
