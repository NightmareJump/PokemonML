using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using TensorFlowLite; 

public class PokemonClassifier : MonoBehaviour
{
    [Header("Model Settings")]
    public TextAsset tfliteModel;         
    public Vector2Int inputImageSize = new Vector2Int(224, 224);

    [Header("Camera")]
    public RawImage cameraPreview;     
    private WebCamTexture webCamTexture;


    private Interpreter interpreter;
    private float[,,] inputBuffer;        
    private float[] outputBuffer;         
    private string[] classNames = { "bulbasaur", "charmander", "squirtle" }; 


    void Start()
    {

        webCamTexture = new WebCamTexture();
        cameraPreview.texture = webCamTexture;
        webCamTexture.Play();


        var options = new InterpreterOptions() { threads = 2 };
        interpreter = new Interpreter(tfliteModel.bytes, options);

        interpreter.ResizeInputTensor(0, new int[] {1, inputImageSize.y, inputImageSize.x, 3});
        interpreter.AllocateTensors();

        inputBuffer = new float[inputImageSize.y, inputImageSize.x, 3];
        outputBuffer = new float[classNames.Length];
    }

    void Update()
    {
        if (webCamTexture.didUpdateThisFrame)
        {
            TextureToTensor(webCamTexture, ref inputBuffer);
            interpreter.SetInputTensorData(0, inputBuffer);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputBuffer);

            int maxIndex = 0;
            float maxProb = 0f;
            for(int i=0; i<outputBuffer.Length; i++)
            {
                if(outputBuffer[i] > maxProb)
                {
                    maxProb = outputBuffer[i];
                    maxIndex = i;
                }
            }

            string predictedPokemon = classNames[maxIndex];
            Debug.Log($"Detected Pokemon: {predictedPokemon} (prob = {maxProb})");

        }
    }


    private void TextureToTensor(WebCamTexture tex, ref float[,,] input)
    {

        Texture2D temp = new Texture2D(tex.width, tex.height, TextureFormat.RGB24, false);
        temp.SetPixels(tex.GetPixels());
        temp.Apply();

        Color[] resizedPixels = ScaleTexture(temp, inputImageSize.x, inputImageSize.y).GetPixels();

        for (int i = 0; i < resizedPixels.Length; i++)
        {
            int x = i % inputImageSize.x;
            int y = i / inputImageSize.x;
            input[y, x, 0] = resizedPixels[i].r; // R
            input[y, x, 1] = resizedPixels[i].g; // G
            input[y, x, 2] = resizedPixels[i].b; // B
        }

        Destroy(temp);
    }

    private Texture2D ScaleTexture(Texture2D source, int targetWidth, int targetHeight)
    {

        Texture2D result = new Texture2D(targetWidth, targetHeight, source.format, false);
        float incX = (1.0f / (float)targetWidth);
        float incY = (1.0f / (float)targetHeight);

        for (int i = 0; i < targetHeight; i++)
        {
            for (int j = 0; j < targetWidth; j++)
            {
                Color newColor = source.GetPixelBilinear((float)j * incX, (float)i * incY);
                result.SetPixel(j, i, newColor);
            }
        }
        result.Apply();
        return result;
    }

    private void OnDestroy()
    {
        if (interpreter != null) interpreter.Dispose();
    }
}
