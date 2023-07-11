using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.DataModel;
using UnityEngine.Perception.GroundTruth.Labelers;

public class StorageEndpoint : IConsumerEndpoint
{
    // Boolean flag indicating whether annotations for all entities are processed
    // up-to-date; false at the beginning, and during processing frames
    [HideInInspector]
    public bool annotationsUpToDate;
    
    // Storage implemented as a nested dictionary; outer mapping from PerceptionCamera ID,
    // to inner mapping from instance ID to annotation
    public Dictionary<string, Dictionary<string, BoundingBox>> boxStorage;
    public Dictionary<string, Dictionary<string, Color32>> maskStorage;

    // Segmentation PNG image buffer
    public Color32[] segMap;

    public string description => "Endpoint that stores latest annotations instead of writing out to disk";

    private Texture2D _segMap;
    
    public object Clone()
    {
        return new StorageEndpoint();
    }

    public bool IsValid(out string errorMessage)
    {
        // No reason for this endpoint to be invalid in any sense
        errorMessage = "Always happy";
        return true;
    }

    public void SimulationStarted(SimulationMetadata metadata)
    {
        // Initialize annotation storage
        boxStorage = new Dictionary<string, Dictionary<string, BoundingBox>>();
        maskStorage = new Dictionary<string, Dictionary<string, Color32>>();
        
        // Initialize segmentation map texture
        _segMap = new Texture2D(2, 2);
    }

    public void SensorRegistered(SensorDefinition sensor)
    {
        // Do nothing
    }

    public void AnnotationRegistered(AnnotationDefinition annotationDefinition)
    {
        // Do nothing
    }

    public void MetricRegistered(MetricDefinition metricDefinition)
    {
        // Do nothing
    }

    public void FrameGenerated(Frame frame)
    {
        if (annotationsUpToDate) return;

        // Clear current storage
        boxStorage.Clear();
        maskStorage.Clear();

        // Retrieve the labeler, then the annotation lists
        var labeler = frame.sensors.OfType<RgbSensor>().Single();
        var boxAnnotations =
            labeler.annotations.OfType<BoundingBoxAnnotation>().Single();
        var maskAnnotations =
            labeler.annotations.OfType<InstanceSegmentationAnnotation>().Single();

        // Instance segmentation map of whole scene view
        _segMap.LoadImage(maskAnnotations.buffer);
        segMap = _segMap.GetPixels32();

        // Initialize inner mapping if not already initialized
        if (!boxStorage.ContainsKey(labeler.id))
            boxStorage[labeler.id] = new Dictionary<string, BoundingBox>();
        if (!maskStorage.ContainsKey(labeler.id))
            maskStorage[labeler.id] = new Dictionary<string, Color32>();

        // Store the annotations
        foreach (var ann in boxAnnotations.boxes)
            boxStorage[labeler.id][$"ent_{ann.instanceId}"] = ann;
        foreach (var ann in maskAnnotations.instances)
            maskStorage[labeler.id][$"ent_{ann.instanceId}"] = ann.color;

        annotationsUpToDate = true;
    }

    public void SimulationCompleted(SimulationMetadata metadata)
    {
        // Release elements, clear storage
        maskStorage.Clear();
    }

    public (string, int) ResumeSimulationFromCrash(int maxFrameCount)
    {
        // Do nothing
        return (string.Empty, 0);
    }
}
