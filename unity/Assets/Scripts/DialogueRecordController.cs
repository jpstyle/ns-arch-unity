using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.UIElements.Experimental;

public class DialogueRecordController
{
    Label RecordContent;

    // This function retrieves a reference to the label element inside
    // the UI element
    public void SetVisualElement(VisualElement visualElement)
    {
        RecordContent = visualElement.Q<Label>("record");

        // Register LinkTagEvent listener
        RecordContent.RegisterCallback<PointerDownLinkTagEvent>(FooMethod);
    }

    // This function receives data of a single dialogue record consisting
    // of speaker info & utterance content (maybe link data designating
    // environmental entities, referenced by demonstratives?)
    public void SetRecordData(RecordData recordData)
    {
        // Color-code the utterance (via rich text tags) according to the
        // speaker info
        switch (recordData.speaker)
        {
            case "Teacher":
            {
                RecordContent.text = $"<color=#FFB6C1>{recordData.speaker}</color>> {recordData.utterance}";
                break;
            }
            case "Student":
            {
                RecordContent.text = $"<color=#ADD8E6>{recordData.speaker}</color>> {recordData.utterance}";
                break;
            }
            case "System":
            {
                RecordContent.text = $"<color=#808080>{recordData.speaker}> {recordData.utterance}</color>";
                break;
            }
            default:
            {
                Debug.Log("Unrecognized speaker type");
                break;
            }
        }
    }

    public void FooMethod(PointerDownLinkTagEvent evt)
    {
        Debug.Log($"{evt.linkID} {evt.linkText}");
    }
}
