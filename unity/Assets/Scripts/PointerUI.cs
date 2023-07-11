using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public class PointerUI : MonoBehaviour
{
    // Highlights object with smallest bounding box + segmentation mask on hover;
    // defined per display

    // Associate with this dialogue UI instance; demonstrative pointing will add
    // to the current input data
    public DialogueUI dialogueUI;

    // Int id of display this pointer UI is associated with
    private int _displayId;

    // List of EnvEntity instances
    private List<EnvEntity> _envEntities;

    // 'Currently' focused EnvEntity instance (hence the associated GameObject)
    private EnvEntity _currentFocus;

    // UI element references
    private VisualElement _screenCover;
    private VisualElement _pointerRect;
    private Label _targetNameLabel;

    private void OnEnable()
    {
        // The UXML is already instantiated by the UIDocument component
        var uiDocument = GetComponent<UIDocument>();
        var root = uiDocument.rootVisualElement;

        _displayId = uiDocument.panelSettings.targetDisplay;

        // VisualElement for detecting mouse movements
        _screenCover = root.Q<VisualElement>("ScreenCover");

        // VisualElement for indicating current focus
        _pointerRect = root.Q<VisualElement>("PointerRect");

        // Label indicating current focus object by name
        _targetNameLabel = _pointerRect.Q<Label>("TargetName");

        // Find and store all existing GameObjects attached with a EnvEntity
        // component (and thus tasked to compute their own per-display screen AABBs)
        _envEntities = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None).ToList();

        _currentFocus = null;

        // Register mousemove event callback to both screenCover and pointerRect (latter
        // needed since it lies on top of screenCover when active)
        _screenCover.RegisterCallback<MouseMoveEvent>(HighlightObjectOnHover);
        _pointerRect.RegisterCallback<MouseMoveEvent>(HighlightObjectOnHover);

        // Register click event callback to pointerRect
        _pointerRect.RegisterCallback<ClickEvent>(PointWithDemonstrative);
    }

    private void HighlightObjectOnHover(MouseMoveEvent evt)
    {
        // Current mouse position in screen coordinate
        var currentPosition = evt.mousePosition;

        // Find boxes that the mouse is currently hovering over
        var entitiesHovering = _envEntities.FindAll(
            ent => ent.masks.Count > 0 && ent.boxes[_displayId].Contains(currentPosition)
        );

        EnvEntity newFocus = null;
        if (entitiesHovering.Count > 0)
        {
            // Select the box with the smallest area on which the mouse is hovering
            var minArea = float.MaxValue;

            foreach (var ent in entitiesHovering)
            {
                var box = ent.boxes[_displayId];
                var boxArea = box.width * box.height;

                if (boxArea < minArea)
                {
                    // Store the currently 'focused' object box
                    newFocus = ent;
                    minArea = boxArea;
                }
            }
        }

        // Update focus -- only if required
        if (!ReferenceEquals(newFocus, _currentFocus))
        {
            _currentFocus = newFocus;

            // Update UI rendering
            if (_currentFocus is null)
                RemoveHighlight();
            else
                HighlightEnt(_currentFocus);
        }
    }

    public void AddEnvEntity(EnvEntity ent)
    {
        _envEntities.Add(ent);
    }

    public void DelEnvEntity(EnvEntity ent)
    {
        _envEntities.Remove(ent);
    }

    public void HighlightEnt(EnvEntity ent)
    {
        // Highlight the target ent by setting display to Flex and changing style
        _pointerRect.style.display = DisplayStyle.Flex;

        var highlightBox = ent.boxes[_displayId];
        var bx = (int) highlightBox.x;
        var by = (int) highlightBox.y;
        var bw = (int) highlightBox.width;
        var bh = (int) highlightBox.height;
        _pointerRect.style.left = new StyleLength(bx);
        _pointerRect.style.top = new StyleLength(by);
        _pointerRect.style.width = new StyleLength(bw);
        _pointerRect.style.height = new StyleLength(bh);

        _targetNameLabel.text = ent.gameObject.name;
    }

    public void RemoveHighlight()
    {
        // Set display of pointerRect to None
        _pointerRect.style.display = DisplayStyle.None;
    }

    private void PointWithDemonstrative(ClickEvent evt)
    {
        if (_currentFocus is null) return;

        // 'Point' at an object by associating it with a demonstrative newly added
        // to the input text field in dialogue UI
        dialogueUI.AddDemonstrativeWithReference(_currentFocus.uid);
    }
}
