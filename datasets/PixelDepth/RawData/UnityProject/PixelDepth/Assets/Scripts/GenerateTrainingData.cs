using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GenerateTrainingData : MonoBehaviour {
   // Use this for initialization
   void Start()
   {
      var worker = new WorkerThread();

      worker.StartThread();
   }
}

public class WorkerThread : MonoBehaviour
{
   private class LabelledItem
   {
      public GameObject gameObject;
      public int label;

      public LabelledItem(GameObject gameObject, int label)
      {
         this.gameObject = gameObject;
         this.label = label;
      }
   }

   string trainingFolderLocation = "c:/temp/training/";
   List<GameObject> visibleItems = new List<GameObject>();
   List<LabelledItem> labelledItems;
   System.Random rand = new System.Random();

   TimeSpan sleepTime = new TimeSpan(0, 0, 0);
   TimeSpan objectPlacementTime = new TimeSpan(0, 0, 0);
   TimeSpan labelCreationTime = new TimeSpan(0, 0, 0);
   TimeSpan fileSavingTime = new TimeSpan(0, 0, 0);
   TimeSpan screenShotTime = new TimeSpan(0, 0, 0);

   // Use this for initialization
   public void StartThread()
   {
      labelledItems = new List<LabelledItem>();
      for (int i = 0; i < 20; i++)
      {
         var labelledItem = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
         var collider = labelledItem.GetComponent<Collider>();
         Destroy(collider);
         labelledItem.AddComponent<MeshCollider>();
         labelledItems.Add(new LabelledItem(labelledItem, 1));
         visibleItems.Add(labelledItem);
      }

      for (int i = 0; i < 20; i++)
      {
         var labelledItem = GameObject.CreatePrimitive(PrimitiveType.Quad);
         var collider = labelledItem.GetComponent<Collider>();
         Destroy(collider);
         labelledItem.AddComponent<MeshCollider>();
         labelledItems.Add(new LabelledItem(labelledItem, 2));
         visibleItems.Add(labelledItem);
      }

      var startTime = DateTime.Now;
      while (true)
      {
         var now = DateTime.Now;
         while (Directory.GetFiles(trainingFolderLocation, "*.dat", SearchOption.TopDirectoryOnly).Length > 1000)
         {
            System.Threading.Thread.Sleep(500);
         }
         sleepTime += DateTime.Now - now;

         now = DateTime.Now;
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
         objectPlacementTime += DateTime.Now - now;

         var guid = Guid.NewGuid();
         foreach (var camera in Camera.allCameras)
         {
            TakeScreenshot(camera, guid);
         }

         GenerateSemanticSegmentationTable(guid);

         using (var tw = new StreamWriter(@"c:\temp\diagnostics.txt", false))
         {
            tw.WriteLine("Sleep time: " + this.sleepTime.TotalSeconds + " s");
            tw.WriteLine("objectPlacementTime: " + this.objectPlacementTime.TotalSeconds + " s");
            tw.WriteLine("labelCreationTime: " + this.labelCreationTime.TotalSeconds + " s");
            tw.WriteLine("fileSavingTime: " + this.fileSavingTime.TotalSeconds + " s");
            tw.WriteLine("screenShotTime: " + this.screenShotTime.TotalSeconds + " s");
            tw.Flush();
            tw.Close();
         }
      }
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

   private void GenerateSemanticSegmentationTable(Guid id)
   {
      var startTime = DateTime.Now;
      // Generate semantic segmentation table
      int width = Camera.allCameras[0].pixelWidth;
      int height = Camera.allCameras[0].pixelHeight;
      byte[] semanticSegmentationTable = new byte[width * height];
      byte label0 = Convert.ToByte('0');
      byte label1 = Convert.ToByte('1');
      byte deliminator = Convert.ToByte(' ');
      int arrPos = 0;
      for (int row = height - 1; row >= 0; row--)
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
               foreach (var labelledItem in labelledItems)
               {
                  if (hit.collider.gameObject == labelledItem.gameObject)
                  {
                     semanticSegmentationTable[arrPos] = (byte)labelledItem.label;
                     break;
                  }
               }
            }
            arrPos += 1;
         }
      }
      this.labelCreationTime += DateTime.Now - startTime;

      startTime = DateTime.Now;
      File.WriteAllBytes(trainingFolderLocation + "/" + id + "_labels.xxx", semanticSegmentationTable);
      File.Move(trainingFolderLocation + "/" + id + "_labels.xxx", trainingFolderLocation + "/" + id + "_labels.dat");
      this.fileSavingTime += DateTime.Now - startTime;
   }

   private void TakeScreenshot(Camera camera, Guid id)
   {
      var startTime = DateTime.Now;
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
      this.screenShotTime += DateTime.Now - startTime;

      startTime = DateTime.Now;
      System.IO.File.WriteAllBytes(filename, screenShot.EncodeToJPG());
      var bytes = screenShot.GetRawTextureData();
      this.fileSavingTime += DateTime.Now - startTime;
   }
}