#include <Arduino.h>
#include <math.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/core/api/error_reporter.h>
#include <PDM.h> 
#include <arm_math.h>
#include "dvector_model_data.h"
#include "constants.h"

#define NUM_MEL_FILTERS    40
#define NUM_MFCC_COEFFS    13
#define SAMPLE_RATE        16000
#define AUDIO_LENGTH       16000   // 1 second
#define FRAME_LEN          512     // FFT length
#define NUM_FRAMES         32
#define FRAME_STEP         (AUDIO_LENGTH / NUM_FRAMES)  // 400
#define PDM_BUFFER_SIZE    512
#define PAD_LEN            256
#define THRESHOLD_UNKNOWN  0.5  // Minimum cosine similarity to accept

const int D_VECTOR_SIZE = 32;

volatile bool      audioReady = false;
volatile int      samplesRead = 0;
volatile int16_t  audioBuffer[AUDIO_LENGTH]; 
volatile int      pdmBytesAvailable = 0;
int16_t    pdmBuffer[PDM_BUFFER_SIZE / 2]; 
float32_t  audioNorm[AUDIO_LENGTH]; 
float32_t  hammingWindow[FRAME_LEN];
float32_t  mfcc_out[NUM_FRAMES][NUM_MFCC_COEFFS]; 



const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
constexpr int TENSOR_ARENA_SIZE = 80 * 1024; // ~80KB
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


void onPDMdata() {
  pdmBytesAvailable = PDM.read((uint8_t*)pdmBuffer, PDM_BUFFER_SIZE);
}

void initHamming() {
  for (int i = 0; i < FRAME_LEN; i++) {
    hammingWindow[i] = 0.54 - 0.46 * cosf(2.0 * PI * i / (FRAME_LEN - 1));
  }
  Serial.println("Hamming filter initialized!");
}

float32_t get_padded_sample(int pos, float32_t* audio) {     //..4, 1234, 1.. 
    if (pos < PAD_LEN) {
        return audio[PAD_LEN - 1 - pos];  // Left reflection
    } 
    else if (pos < PAD_LEN + AUDIO_LENGTH) {
        return audio[pos - PAD_LEN];       // Main audio
    } 
    else {
        return audio[AUDIO_LENGTH - 2 - (pos - (PAD_LEN + AUDIO_LENGTH))];  // Right reflection
    }
}

void mfcc(){
  arm_rfft_fast_instance_f32 fft;
  arm_rfft_fast_init_f32(&fft, FRAME_LEN);

  audioNorm[0] = (float32_t)audioBuffer[0] / 32768.0f;
  for (int i = 1; i < AUDIO_LENGTH; i++) {
      audioNorm[i] = ((float32_t)audioBuffer[i] * (1.0f / 32768.0f)) - 0.97f * ((float32_t)audioBuffer[i-1] * (1.0f / 32768.0f));
  }
    
  for (int f = 0; f < NUM_FRAMES; f++) {
    float32_t frame[FRAME_LEN];
    int start = f * FRAME_STEP;

    //Apply Hamming window
    for (int i = 0; i < FRAME_LEN; i++) {
      frame[i] = (get_padded_sample(i + start, audioNorm)) * hammingWindow[i];  
    }

    //Perform FFT
    float32_t spectrum[FRAME_LEN];
    memset(spectrum, 0, sizeof(spectrum));
    arm_rfft_fast_f32(&fft, frame, spectrum, 0);

    //Compute Power
    int bins = FRAME_LEN / 2 + 1;
    float32_t power[bins];
    power[0] = spectrum[0] * spectrum[0];  // DC bin
    power[bins - 1] = spectrum[1] * spectrum[1];  // Nyquist
    for (int i = 1; i < bins - 1; i++) {
      float re = spectrum[2 * i];
      float im = spectrum[2 * i + 1];
      power[i] = re * re + im * im;
    }
    
    //Apply Mel filterbank
    float32_t melE[NUM_MEL_FILTERS];
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
      float sum = 0;
      for (int i = 0; i < bins; i++) sum += melFilterBank[m][i] * power[i];
      melE[m] = logf(max(sum, 1e-10));
    }

    // DCT to MFCC
    for (int c = 0; c < NUM_MFCC_COEFFS; c++) {
      float val = 0;
      for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        val += dctMatrix[c][m] * melE[m];
      }
      mfcc_out[f][c] = val;
    }
  }
}

float cosineSimilarity(const float* vecA, const float* vecB, int size) {
  float dot = 0.0, normA = 0.0, normB = 0.0;
  for (int i = 0; i < size; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (sqrt(normA) * sqrt(normB));
}

void setup() {
  Serial.begin(9600);
  delay(2000);
  //while (!Serial);
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);

  initHamming();


  model = tflite::GetModel(dvector_model_tflite);
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddMul();  
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape(); 
  micro_op_resolver.AddFullyConnected(); 
  micro_op_resolver.AddL2Normalization();
  static tflite::MicroInterpreter static_interpreter(
      model, 
      micro_op_resolver,
      tensor_arena,
      TENSOR_ARENA_SIZE
  );
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("TF Allocation FAILED!");
    Serial.print("Required memory: ");
    Serial.println(interpreter->arena_used_bytes());
    while(1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) { // 1 channel
    Serial.println("PDM MIC FAIL!");
    while (1);
  }
  PDM.setGain(127); //Gain set to maximum

  Serial.println("System ready");
}

void loop() {
  //digitalWrite(LEDR, LOW); 
  delay(100); 
  
  //Handle PDM buffer
  if (pdmBytesAvailable > 0 && !audioReady) {
    int samplesToCopy = pdmBytesAvailable / sizeof(int16_t);

    for (int i = 0; i < samplesToCopy && samplesRead < AUDIO_LENGTH; i++) {
      audioBuffer[samplesRead++] = pdmBuffer[i];
    }
    pdmBytesAvailable = 0;

    if (samplesRead >= AUDIO_LENGTH) {
      audioReady = true;
    }
  }
  
  if (audioReady) {
    //digitalWrite(LEDR, LOW); 
    Serial.println("Audio captured");
 
    mfcc();
    
    Serial.println("MFCC computed");

    for(int i = 0; i<NUM_FRAMES; i++){
      for(int j = 0; j<NUM_MFCC_COEFFS; j++){
        input->data.f[i * NUM_MFCC_COEFFS + j] =  mfcc_out[i][j];   //mfccExample[i][j];
        //Serial.print(mfcc_out[i][j]);
        //Serial.print(", ");
      }
      //Serial.println();
    }
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Inference FAILED!");
      audioReady = false;
      samplesRead = 0;
      return;
    }

    float* current_d_vector = output->data.f;

    float sim_me = cosineSimilarity(current_d_vector, REF_D_VECTOR_ME, D_VECTOR_SIZE);

    Serial.print("\nSimilarity: ");
    Serial.print(sim_me);
    Serial.print(" - Result: ");
    Serial.println(sim_me > THRESHOLD_UNKNOWN ? "ME" : "NOTME");
    
    for(int i=0; i<32; i++){
       Serial.print(current_d_vector[i]);
       Serial.print(",");
    }
    Serial.println();
    
    audioReady = false;
    samplesRead = 0;
    digitalWrite(LEDR, HIGH);
    
  }
}

