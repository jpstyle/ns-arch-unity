using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.DataModel;
using UnityEngine.Perception.GroundTruth.Labelers;

public class StorageEndpoint : IConsumerEndpoint
{
    // Boolean flag indicating whether boxes for all entities are processed
    // up-to-date; false at the beginning, and during processing frames
    [HideInInspector]
    public bool boxesUpToDate;
    
    // Storage implemented as a nested dictionary; outer mapping from PerceptionCamera ID,
    // to inner mapping from instance ID to bounding box
    public Dictionary<string, Dictionary<string, BoundingBox>> boxStorage;

    public string description => "Endpoint that stores latest annotations instead of writing out to disk";

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
        // Retrieve the bounding box labeler, then the annotation list
        var boxLabeler = frame.sensors.OfType<RgbSensor>().Single();
        var boxAnnotations = boxLabeler.annotations.OfType<BoundingBoxAnnotation>().Single();

        // Initialize inner mapping if not already initialized
        if (!boxStorage.ContainsKey(boxLabeler.id))
            boxStorage[boxLabeler.id] = new Dictionary<string, BoundingBox>();
        
        // Store the box annotations
        foreach (var ann in boxAnnotations.boxes)
            boxStorage[boxLabeler.id][$"ent_{ann.instanceId}"] = ann;

        boxesUpToDate = true;
    }

    public void SimulationCompleted(SimulationMetadata metadata)
    {
        // Release elements, clear storage
        boxStorage.Clear();
    }

    public (string, int) ResumeSimulationFromCrash(int maxFrameCount)
    {
        // Do nothing
        return (string.Empty, 0);
    }
}
