using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public class PointerUI : MonoBehaviour
{
    // Highlights object with smallest bounding box on hover; defined per display

    // Associate with this dialogue UI instance; demonstrative pointing will add
    // to the current input data
    public DialogueUI dialogueUI;

    // Int id of display this pointer UI is associated with
    int displayId;

    // List of EnvEntity instances
    List<EnvEntity> boxExtractors;

    // 'Currently' focused EnvEntity instance (hence the associated GameObject)
    EnvEntity currentFocus;

    // UI element references
    VisualElement screenCover;
    VisualElement pointerRect;
    Label targetNameLabel;

    void OnEnable()
    {
        // The UXML is already instantiated by the UIDocument component
        UIDocument uiDocument = GetComponent<UIDocument>();
        VisualElement root = uiDocument.rootVisualElement;

        displayId = uiDocument.panelSettings.targetDisplay;

        // VisualElement for detecting mouse movements
        screenCover = root.Q<VisualElement>("ScreenCover");

        // VisualElement for indicating current focus
        pointerRect = root.Q<VisualElement>("PointerRect");

        // Label indicating current focus object by name
        targetNameLabel = pointerRect.Q<Label>("TargetName");

        // Find and store all existing GameObjects attached with a EnvEntity
        // component (and thus tasked to compute their own per-display screen AABBs)
        boxExtractors = Object.FindObjectsByType<EnvEntity>(FindObjectsSortMode.None).ToList();

        currentFocus = null;

        // Register mousemove event callback to both screenCover and pointerRect (latter
        // needed since it lies on top of screenCover when active)
        screenCover.RegisterCallback<MouseMoveEvent>(HighlightObjectOnHover);
        pointerRect.RegisterCallback<MouseMoveEvent>(HighlightObjectOnHover);

        // Register click event callback to pointerRect
        pointerRect.RegisterCallback<ClickEvent>(PointWithDemonstrative);
    }

    public void HighlightObjectOnHover(MouseMoveEvent evt)
    {
        // Current mouse position in screen coordinate
        Vector2 currentPosition = evt.mousePosition;

        // Find boxes that the mouse is currently hovering over
        List<EnvEntity> boxExtractorsHovering = boxExtractors.FindAll(
            extr => extr.boxes[displayId].Contains(currentPosition)
        );

        EnvEntity newFocus = null;
        if (boxExtractorsHovering.Count > 0)
        {
            // Select the box with the smallest area on which the mouse is hovering
            float minArea = float.MaxValue;

            foreach (EnvEntity ent in boxExtractorsHovering)
            {
                Rect box = ent.boxes[displayId];
                float boxArea = box.width * box.height;

                if (boxArea < minArea)
                {
                    // Store the currently 'focused' object box
                    newFocus = ent;
                    minArea = boxArea;
                }
            }
        }
        else
        {
            newFocus = null;
        }

        // Update focus -- only if required
        if (!ReferenceEquals(newFocus, currentFocus))
        {
            currentFocus = newFocus;

            // Update UI rendering
            if (currentFocus is null)
            {
                // Set display of pointerRect to None
                pointerRect.style.display = DisplayStyle.None;
            }
            else
            {
                // Highlight the current focus by setting display to Flex and changing style
                pointerRect.style.display = DisplayStyle.Flex;

                Rect highlightBox = currentFocus.boxes[displayId];
                pointerRect.style.left = new StyleLength(highlightBox.x);
                pointerRect.style.top = new StyleLength(highlightBox.y);
                pointerRect.style.width = new StyleLength(highlightBox.width);
                pointerRect.style.height = new StyleLength(highlightBox.height);

                targetNameLabel.text = currentFocus.gameObject.name;
            }
        }
    }

    public void PointWithDemonstrative(ClickEvent evt)
    {
        // 'Point' at an object by associating it with a demonstrative newly added
        // to the input text field in dialogue UI
        if (currentFocus is not null)
            Debug.Log(currentFocus.gameObject.name);
    }
}
