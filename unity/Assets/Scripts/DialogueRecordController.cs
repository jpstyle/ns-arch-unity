using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.UIElements.Experimental;

public class DialogueRecordController
{
    private Label _recordContent;

    // This function retrieves a reference to the label element inside
    // the UI element
    public void SetVisualElement(VisualElement recordContainer)
    {
        _recordContent = recordContainer.Q<Label>("record");

        // Register LinkTagEvent listener
        _recordContent.RegisterCallback<PointerOverLinkTagEvent>(HighlightReference);
        _recordContent.RegisterCallback<PointerOutLinkTagEvent>(RemoveReferenceHighlight);
    }

    // This function receives data of a single dialogue record consisting
    // of speaker info, utterance content and demonstrative references
    public void SetRecordData(RecordData recordData)
    {
        // To insert link tags for handling demonstrative references info, find text segments
        // divided by <link> or </link>
        var segmentRanges = new Dictionary<int, int>
        {
            [0] = recordData.utterance.Length
        };

        // Sorted list of substring positions corresponding to each demonstrative pronoun
        var demRefRanges = recordData.demonstrativeReferences.Keys?.ToList();
        demRefRanges ??= new List<(int, int)>();
        demRefRanges.Sort();       // Sort ascending order

        var lastSegmentEnd = 0;
        foreach (var (start, end) in demRefRanges)
        {
            if (lastSegmentEnd != start)
                segmentRanges[lastSegmentEnd] = start;      // Add only if Length!=0
            segmentRanges[start] = end;
            lastSegmentEnd = end;
        }

        // Add possible last segment if Length!=0
        if (lastSegmentEnd != recordData.utterance.Length)
            segmentRanges[lastSegmentEnd] = recordData.utterance.Length;

        // Build utterance text with demRef links inserted
        var utteranceWithLink = "";
        foreach (var (start, end) in segmentRanges)
        {
            var utterancePiece = recordData.utterance[start..end];
            if (recordData.demonstrativeReferences.ContainsKey((start, end)))
            {
                // Matching reference found, add text wrapped with link tags
                var refEnt = recordData.demonstrativeReferences[(start, end)];
                utteranceWithLink += $"<color=#FF9600><link=\"{refEnt}\">{utterancePiece}</link></color>";
            }
            else
            {
                // Matching reference not found, simply add text
                utteranceWithLink += utterancePiece;
            }
        }

        // Color-code the utterance according to the speaker info
        switch (recordData.speaker)
        {
            case "Teacher":
            {
                _recordContent.text = $"<color=#FFB6C1>{recordData.speaker}</color>> {utteranceWithLink}";
                break;
            }
            case "Student":
            {
                _recordContent.text = $"<color=#ADD8E6>{recordData.speaker}</color>> {utteranceWithLink}";
                break;
            }
            case "System":
            {
                _recordContent.text = $"<color=#808080>{recordData.speaker}> {utteranceWithLink}</color>";
                break;
            }
            default:
            {
                Debug.Log("Unrecognized speaker type");
                break;
            }
        }
    }

    private static void HighlightReference(PointerOverLinkTagEvent evt)
    {
        var ent = EnvEntity.FindByUid(evt.linkID);
        
        var pointerUIs = Object.FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.HighlightEnt(ent);
    }

    private static void RemoveReferenceHighlight(PointerOutLinkTagEvent evt)
    {
        var pointerUIs = Object.FindObjectsByType<PointerUI>(FindObjectsSortMode.None);
        foreach (var pUI in pointerUIs)
            pUI.RemoveHighlight();
    }
}
