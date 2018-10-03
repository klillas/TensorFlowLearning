using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GenerateTrainingData : MonoBehaviour {
   string trainingFolderLocation = "c:/temp/training/";

   List<GameObject> visibleItems = new List<GameObject>();

	// Use this for initialization
	void Start () {
      int examplesToCreate = 50;
      var rand = new System.Random();
      var prefab = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
      // prefab.AddComponent<MeshCollider>();
      visibleItems.Add(prefab);

      var startTime = DateTime.Now;
      for (int i = 0; i < examplesToCreate; i++)
      {
         float zPos = (float)(rand.NextDouble() * 100.0);
         prefab.transform.position = new Vector3(0, 1, zPos);

         foreach (var camera in Camera.allCameras)
         {
            TakeScreenshot(camera, i);
         }

         GenerateSemanticSegmentationTable(i);
      }
      var endTime = DateTime.Now;
      var timespan = endTime - startTime;
      print("Time per example: " + timespan.Milliseconds / examplesToCreate + "ms");
   }

   private void GenerateSemanticSegmentationTable(int id)
   {
      // Generate semantic segmentation table
      int width = Camera.allCameras[0].pixelWidth;
      int height = Camera.allCameras[0].pixelHeight;
      byte[] semanticSegmentationTable = new byte[width * height];
      byte label0 = Convert.ToByte('0');
      byte label1 = Convert.ToByte('1');
      byte deliminator = Convert.ToByte(' ');
      int arrPos = 0;
      for (int row = 0; row < height; row++)
      {
         for (int column = 0; column < width; column++)
         {
            Ray ray = Camera.allCameras[0].ScreenPointToRay(new Vector3(column, row, 0));
            RaycastHit hit;
            Physics.Raycast(ray, out hit);
            if (hit.collider == null)
            {
               semanticSegmentationTable[arrPos] = 0;
            }
            else
            {
               // TODO: Add type information, but for now go with static 1
               semanticSegmentationTable[arrPos] = 1;
            }
            // semanticSegmentationTable[arrPos + 1] = deliminator;
            arrPos += 1;
         }
      }

      File.WriteAllBytes(trainingFolderLocation + "/" + id + "_labels.dat", semanticSegmentationTable);
   }

   private void TakeScreenshot(Camera camera, int id)
   {
      var width = camera.pixelWidth;
      var height = camera.pixelHeight;
      var filename = string.Format("c:/temp/training/" + id + "_" + camera.name + ".jpg");

      RenderTexture rt = new RenderTexture(width, height, 24);
      RenderTexture.active = rt;
      camera.targetTexture = rt;

      camera.Render();

      Texture2D screenShot = new Texture2D(width, height);
      screenShot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
      screenShot.Apply();

      RenderTexture.active = null;
      camera.targetTexture = null;
      DestroyImmediate(rt);

      System.IO.File.WriteAllBytes(filename, screenShot.EncodeToJPG());
      var bytes = screenShot.GetRawTextureData();
      Debug.Log(string.Format("Took screenshot to: {0}", filename));
      Debug.Log("Width: " + width + ", Height: " + height);
   }
	
	// Update is called once per frame
	void Update () {
		
	}
}
