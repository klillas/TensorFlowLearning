using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GenerateTrainingData : MonoBehaviour {
   string trainingFolderLocation = "c:/temp/training/";
   List<GameObject> visibleItems = new List<GameObject>();
   List<GameObject> labelledItems;
   System.Random rand = new System.Random();

   // Use this for initialization
   void Start () {
      labelledItems = new List<GameObject>();
      for (int i = 0; i < 30; i++)
      {
         var labelledItem = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
         var collider = labelledItem.GetComponent<Collider>();
         Destroy(collider);
         labelledItem.AddComponent<MeshCollider>();
         labelledItems.Add(labelledItem);
         visibleItems.Add(labelledItem);
      }

      /*
      for (int i = 0; i < 10; i++)
      {
         var otherObject = GameObject.CreatePrimitive(PrimitiveType.Quad);
         visibleItems.Add(otherObject);
      }
      */

      var startTime = DateTime.Now;
      for (int i = 10000000; i < int.MaxValue; i++)
      {
         while (Directory.GetFiles(trainingFolderLocation, "*.dat", SearchOption.TopDirectoryOnly).Length > 1000)
         {
            System.Threading.Thread.Sleep(500);
         }

         foreach (var gameObject in visibleItems)
         {
            RandomlyPlaceObjectInCameraView(Camera.allCameras[0], gameObject);
         }

         foreach (var obj in visibleItems)
         {
            if (rand.NextDouble() < 0.5)
            {
               obj.SetActive(false);
            }
            else
            {
               obj.SetActive(true);
            }
         }

         foreach (var camera in Camera.allCameras)
         {
            TakeScreenshot(camera, i);
         }

         GenerateSemanticSegmentationTable(i);
      }
      var endTime = DateTime.Now;
      var timespan = endTime - startTime;
   }

   void DeleteExampleWithId(int id)
   {

   }

   void RandomlyPlaceObjectInCameraView(Camera camera, GameObject gameObject)
   {
      float zPos = (float)(rand.NextDouble() * 10);
      float xPos = (float)(rand.NextDouble() * camera.pixelWidth);
      float yPos = (float)(rand.NextDouble() * camera.pixelHeight);
      var screenPoint = new Vector3(xPos, yPos, zPos + 5);
      var worldPos = camera.ScreenToWorldPoint(screenPoint);
      gameObject.transform.position = worldPos;
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
      int pixelsLabeled = 0;
      for (int row = height - 1; row >= 0; row--)
      {
         for (int column = 0; column < width; column++)
         {
            Ray ray = Camera.allCameras[0].ScreenPointToRay(new Vector3(column, row, 0));
            RaycastHit hit;
            Physics.Raycast(ray, out hit);
            if (hit.collider != null && labelledItems.Contains(hit.collider.gameObject))
            {
               semanticSegmentationTable[arrPos] = 1;
            }
            else
            {
               pixelsLabeled++;
               semanticSegmentationTable[arrPos] = 0;
            }
            // semanticSegmentationTable[arrPos + 1] = deliminator;
            arrPos += 1;
         }
      }

      File.WriteAllBytes(trainingFolderLocation + "/" + id + "_labels.xxx", semanticSegmentationTable);
      File.Move(trainingFolderLocation + "/" + id + "_labels.xxx", trainingFolderLocation + "/" + id + "_labels.dat");
   }

   private void TakeScreenshot(Camera camera, int id)
   {
      var width = camera.pixelWidth;
      var height = camera.pixelHeight;
      var filename = string.Format(trainingFolderLocation + id + "_" + camera.name + ".jpg");

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
