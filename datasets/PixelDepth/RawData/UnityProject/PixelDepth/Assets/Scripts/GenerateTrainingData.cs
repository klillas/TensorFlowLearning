using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class GenerateTrainingData : MonoBehaviour {
   public GameObject ConsoleOutput;
   public List<Material> Materials = new List<Material>();
   public List<GameObject> Lights = new List<GameObject>();
   public GameObject BackWall;
   public GameObject Floor;

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

   private GameObject[] Prefabs;
   string trainingFolderLocation = "c:/temp/training/";
   List<GameObject> visibleItems = new List<GameObject>();
   Dictionary<int, LabelledItem> labelledItems;
   System.Random rand = new System.Random();

   TimeSpan sleepTime = new TimeSpan(0, 0, 0);
   TimeSpan objectPlacementTime = new TimeSpan(0, 0, 0);
   TimeSpan labelCreationTime = new TimeSpan(0, 0, 0);
   TimeSpan fileSavingTime = new TimeSpan(0, 0, 0);
   TimeSpan screenShotTime = new TimeSpan(0, 0, 0);

   DateTime scriptStartTime = DateTime.Now;
   int examplesGenerated = 0;

   List<PrimitiveType> primitiveTypes;

   int width = 256; // or something else
   int height = 192; // or something else

   private void Start()
   {
      bool isFullScreen = false; // should be windowed to run in arbitrary resolution
      int desiredFPS = 60; // or something else
      Screen.SetResolution(width, height, isFullScreen, desiredFPS);

      for (int i = 0; i < 20; i++)
      {
         var go = new GameObject();
         var light = go.AddComponent<Light>();
         Lights.Add(go);
      }

      BackWall = GameObject.CreatePrimitive(PrimitiveType.Cube);
      var collider = BackWall.GetComponent<Collider>();
      Destroy(collider);
      BackWall.AddComponent<MeshCollider>();
      BackWall.transform.localScale = new Vector3(100, 100, 1);

      //Floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
      //Floor.transform.localScale = new Vector3(100, 0.1f, 100);

      Prefabs = Resources.LoadAll<GameObject>("Prefab/");

      primitiveTypes = new List<PrimitiveType>
      {
         PrimitiveType.Capsule,
         PrimitiveType.Cube,
         PrimitiveType.Cylinder
      };

      labelledItems = new Dictionary<int, LabelledItem>();
      for (int i = 0; i < primitiveTypes.Count; i++)
      {
         var item = GameObject.CreatePrimitive(primitiveTypes[i]);
         collider = item.GetComponent<Collider>();
         Destroy(collider);
         item.AddComponent<MeshCollider>();
         labelledItems.Add(item.GetHashCode(), new LabelledItem(item, 0));
         visibleItems.Add(item);
      }

      for (int i = 0; i < Prefabs.Length; i++)
      {
         var item = (GameObject)Instantiate(Prefabs[i], transform.position, transform.rotation);
         //item.AddComponent<MeshRenderer>();
         collider = item.GetComponent<Collider>();
         Destroy(collider);
         item.AddComponent<MeshCollider>();
         labelledItems.Add(item.GetHashCode(), new LabelledItem(item, 0));
         visibleItems.Add(item);
      }

      for (int i = 0; i < 200; i++)
      {
         var labelledItem = GameObject.CreatePrimitive(PrimitiveType.Sphere);
         collider = labelledItem.GetComponent<Collider>();
         Destroy(collider);
         labelledItem.AddComponent<MeshCollider>();
         labelledItems.Add(labelledItem.GetHashCode(), new LabelledItem(labelledItem, 1));
         visibleItems.Add(labelledItem);
      }
   }

   void Update()
   {
      var timeUpdateStarted = DateTime.Now;
      var currentWidth = Camera.allCameras[0].pixelWidth;
      var currentHeight = Camera.allCameras[0].pixelHeight;
      if (currentWidth != width || currentHeight != height)
      {
         // Attempt a fix for the next screen refresh
         bool isFullScreen = false; // should be windowed to run in arbitrary resolution
         int desiredFPS = 60; // or something else
         Screen.SetResolution(width, height, isFullScreen, desiredFPS);

         throw new NotSupportedException("Incorrect resolution on screen, cannot generate correct screenshots." +
            currentWidth + "x" + currentHeight);
      }

      var startTime = DateTime.Now;
      while (true && DateTime.Now - timeUpdateStarted < TimeSpan.FromMilliseconds(250))
      {
         var now = DateTime.Now;
         if (Directory.GetFiles(trainingFolderLocation, "*.dat", SearchOption.TopDirectoryOnly).Length > 1000)
         {
            return;
         }
         sleepTime += DateTime.Now - now;

         for (int i = 0; i < 5; i++)
         {
            now = DateTime.Now;
            foreach (var light in Lights)
            {
               RandomlyPlaceObjectInCameraView(Camera.allCameras[0], light, 1, 20, 3, 3);
               RandomlySetLightProperties(light);

               if (rand.NextDouble() < 0.8)
               {
                  light.SetActive(false);
               }
               else
               {
                  light.SetActive(true);
               }
            }

            foreach (var labelledItem in labelledItems)
            {
               var gameObject = labelledItem.Value.gameObject;
               if (labelledItem.Value.label == 0)
               {
                  RandomlyPlaceObjectInCameraView(Camera.allCameras[0], gameObject, 2, 20, 1, 1);
               }
               else
               {
                  RandomlyPlaceObjectInCameraView(Camera.allCameras[0], gameObject, 1, 20, 1, 1);
               }
               RandomlyAssignMaterialsToObject(gameObject);
               RandomlySetColorToObjectMaterial(gameObject);

               if (rand.NextDouble() > 0.96)
               {
                  gameObject.SetActive(true);
               }
               else
               {
                  gameObject.SetActive(false);
               }
            }

            foreach (var gameObject in labelledItems.Where(x => x.Value.label == 1))
            {
               RandomlyMutateObjectScale(gameObject.Value.gameObject);
            }

            RandomlyPlaceWalls();
            RandomlyAssignMaterialsToObject(BackWall);
            RandomlySetColorToObjectMaterial(BackWall);

            /*
            foreach (var labelledItem in labelledItems.Values)
            {
               labelledItem.gameObject.SetActive(true);
            }
            */
            objectPlacementTime += DateTime.Now - now;

            var guid = Guid.NewGuid();
            foreach (var camera in Camera.allCameras)
            {
               TakeScreenshot(camera, guid);
            }

            GenerateSemanticSegmentationTable(guid);

            examplesGenerated++;

            //var consoleText = this.ConsoleOutput.GetComponent<Text>();
            //string text = "Sleep time: " + this.sleepTime.TotalSeconds + " s\r\n";
            //text += "objectPlacementTime: " + this.objectPlacementTime.TotalSeconds + " s\r\n";
            //text += "labelCreationTime: " + this.labelCreationTime.TotalSeconds + " s\r\n";
            //text += "labelCreationTime: " + this.labelCreationTime.TotalSeconds + " s\r\n";
            //text += "screenShotTime: " + this.screenShotTime.TotalSeconds + " s\r\n";
            //string text = "Examples per second: " + this.examplesGenerated / (DateTime.Now - this.scriptStartTime).TotalSeconds;
            //consoleText.text = text;
         }
      }
   }

   /// <summary>
   /// Randomly places and rotates object
   /// </summary>
   /// <param name="camera">The camera which is used as the view to place the objects</param>
   /// <param name="gameObject">The object to place</param>
   /// <param name="closestZ">The closest Z allowed in meters</param>
   /// <param name="farthestZ">The furthest Z allowed in meters</param>
   /// <param name="xPosMaxPos">The largest x position in camera width %. 1 is the exact width</param>
   /// <param name="yPosMaxPos">The largest y position in camera height. 1 is the exact height </param>
   void RandomlyPlaceObjectInCameraView(Camera camera, GameObject gameObject, float closestZ, float farthestZ, float xPosMaxPos, float yPosMaxPos)
   {
      float zPos = (float)(rand.NextDouble() * farthestZ);
      float xPos = (float)(rand.NextDouble() * camera.pixelWidth * xPosMaxPos) - ((xPosMaxPos - 1f) * camera.pixelWidth * 0.5f);
      float yPos = (float)(rand.NextDouble() * camera.pixelHeight * yPosMaxPos) - ((yPosMaxPos - 1f) * camera.pixelHeight * 0.5f);

      float xRot = (float)(rand.NextDouble() * 360);
      float yRot = (float)(rand.NextDouble() * 360);
      float zRot = (float)(rand.NextDouble() * 360);

      var screenPoint = new Vector3(xPos, yPos, zPos + closestZ);
      var worldPos = camera.ScreenToWorldPoint(screenPoint);
      gameObject.transform.position = worldPos;
      gameObject.transform.eulerAngles = new Vector3(
       xRot,
       yRot,
       zRot
      );
   }

   void RandomlyAssignMaterialsToObject(GameObject gameObject)
   {
      foreach (var renderer in gameObject.GetComponentsInChildren<Renderer>())
      {
         int materialId = (int)(Materials.Count * rand.NextDouble());
         // TODO: Well this is stupid, but it is late and I am too tired to figure out how to avoid it. I assume a math.floor will work but I don't want an exception thrown right now...
         if (materialId == Materials.Count)
         {
            materialId--;
         }

         renderer.material = Materials[materialId];
      }
   }

   void RandomlySetColorToObjectMaterial(GameObject gameObject)
   {
      //Fetch the Renderer from the GameObject
      foreach (var renderer in gameObject.GetComponentsInChildren<Renderer>())
      {
         var mainColor = new Color((float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble());
         var specularShaderColor = new Color((float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble());

         //Set the main Color of the Material to green
         renderer.sharedMaterial.SetColor("_Color", mainColor);

         //Find the Specular shader and change its Color to red
         renderer.sharedMaterial.SetColor("_SpecColor", specularShaderColor);

      }
   }

   void RandomlyMutateObjectScale(GameObject gameObject)
   {
      // TODO: This assume a default scale of 1. Fix it such that it knows the initial scale
      float xScale = (float)(rand.NextDouble() * 0.1 + 1);
      float yScale = (float)(rand.NextDouble() * 0.1 + 1);
      float zScale = (float)(rand.NextDouble() * 0.1 + 1);

      gameObject.transform.localScale = new Vector3(xScale, yScale, zScale);
   }

   void RandomlySetLightProperties(GameObject lightObject)
   {
      var lightComponent = lightObject.GetComponent<Light>();
      lightComponent.intensity = (float) (rand.NextDouble() * 10 + 1);
      lightComponent.range = (float)(rand.NextDouble() * 40 + 1);
      var mainColor = new Color((float)rand.NextDouble(), (float)rand.NextDouble(), (float)rand.NextDouble());
      lightComponent.color = mainColor;
   }

   void RandomlyPlaceWalls()
   {
      float zPos = (float)(rand.NextDouble() * 10 + 10);
      BackWall.transform.position = new Vector3(0, 0, (float)zPos);
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
               int hashCode = hit.collider.gameObject.GetHashCode();
               if (labelledItems.ContainsKey(hashCode))
               {
                  semanticSegmentationTable[arrPos] = (byte)labelledItems[hashCode].label;
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
      this.screenShotTime += DateTime.Now - startTime;

      startTime = DateTime.Now;
      System.IO.File.WriteAllBytes(filename, screenShot.EncodeToJPG());
      var bytes = screenShot.GetRawTextureData();
      this.fileSavingTime += DateTime.Now - startTime;

      Destroy(rt);
      Destroy(screenShot);
   }

}
