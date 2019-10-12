#include <QApplication>
#include "mainwindow.h"

int main(int argc, char* argv[]) {
  //    QApplication a(argc, argv);
  //    MainWindow w;
  //    w.show();

  //    return a.exec();
  tflite::FlatBufferModel model(path_to_model);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  // Resize input tensors, if desired.
  interpreter->AllocateTensors();

  float* input = interpreter->typed_input_tensor<float>(0);
  // Fill `input`.

  interpreter->Invoke();

  float* output = interpreter->typed_output_tensor<float>(0);

  return 0;
}
