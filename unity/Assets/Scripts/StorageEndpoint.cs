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
    public Dictionary<string, Dictionary<string, float[]>> maskStorage;

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
        maskStorage = new Dictionary<string, Dictionary<string, float[]>>();
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
        // Retrieve the labeler, then the annotation list
        var labeler = frame.sensors.OfType<RgbSensor>().Single();
        var maskAnnotations =
            labeler.annotations.OfType<InstanceSegmentationAnnotation>().Single();

        // Instance segmentation map of whole scene view
        var segMap = new Texture2D(2, 2);
        segMap.LoadImage(maskAnnotations.buffer);
        var colorMap = segMap.GetPixels32();

        // Initialize inner mapping if not already initialized
        if (!maskStorage.ContainsKey(labeler.id))
            maskStorage[labeler.id] = new Dictionary<string, float[]>();

        // Store the annotations
        foreach (var ann in maskAnnotations.instances)
        {
            // Color match to get boolean (2d) array, then convert to float
            var msk = colorMap
                .Select(
                    c => c.r == ann.color.r && c.g == ann.color.g && c.b == ann.color.b
                )
                .Select(Convert.ToSingle)
                .ToArray();
            maskStorage[labeler.id][$"ent_{ann.instanceId}"] = msk;
        }

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
