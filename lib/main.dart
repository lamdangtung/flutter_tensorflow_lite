import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as libImage;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Interpreter _interpreter;
  late List<int> _inputShape;
  late List<int> _outputShape;
  final String urlImage =
      "https://dfstudio-d420.kxcdn.com/wordpress/wp-content/uploads/2019/06/digital_camera_photo-980x653.jpg";
  Uint8List? _originalImageBytes;
  Uint8List? _outputImageBytes;

  bool isProcessing = false;

  Future<void> _loadImage() async {
    final imageBytes = await _loadNetworkImage(urlImage);
    setState(() {
      _originalImageBytes = imageBytes;
    });
  }

  Future<void> _run() async {
    if (_originalImageBytes == null) {
      return;
    }
    setState(() {
      isProcessing = true;
    });
    final img = libImage.decodeJpg(_originalImageBytes!);

    if (img == null) {
      return;
    }

    final imgInput = libImage.copyResize(
      img,
      width: _inputShape[1],
      height: _inputShape[2],
    );

    final imageMatrix = List.generate(
      imgInput.height,
      (y) => List.generate(
        imgInput.width,
        (x) {
          final pixel = imgInput.getPixel(x, y);
          // normalize -1 to 1
          return [
            (pixel.r - 127.5) / 127.5,
            (pixel.b - 127.5) / 127.5,
            (pixel.g - 127.5) / 127.5
          ];
        },
      ),
    );

    // Set tensor input [1, 257, 257, 3]
    final input = [imageMatrix];
    // Set tensor output [1, 257, 257, 21]
    final output = [
      List.filled(_outputShape[1],
          List.filled(_outputShape[2], List.filled(_outputShape[3], 0.0)))
    ];
    _interpreter.run(input, output);
    List<List<double>> segmentationMask = extractSegmentationMask(output);
    removeBackground(imgInput, segmentationMask);
  }

  List<List<double>> extractSegmentationMask(
      List<List<List<List<double>>>> output) {
    int height = _outputShape[1];
    int width = _outputShape[2];
    int numClasses = _outputShape[3];

    List<List<double>> mask =
        List.generate(height, (i) => List.generate(width, (j) => 0.0));

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        double maxScore = double.negativeInfinity;
        int maxIndex = 0;
        for (int k = 0; k < numClasses; k++) {
          if (output[0][i][j][k] > maxScore) {
            maxScore = output[0][i][j][k];
            maxIndex = k;
          }
        }
        mask[i][j] = maxIndex.toDouble();
      }
    }

    return mask;
  }

  void removeBackground(libImage.Image originalImage, List<List<double>> mask) {
    // Check if the image and mask dimensions match
    if (originalImage.width != mask.length ||
        originalImage.height != mask[0].length) {
      throw Exception("Image dimensions and mask dimensions do not match");
    }

    // Iterate over each pixel and apply the mask
    for (int i = 0; i < originalImage.width; i++) {
      for (int j = 0; j < originalImage.height; j++) {
        // Get the class index from the mask
        int classIndex = mask[i][j].toInt();

        // Assuming class index 0 is the background class
        if (classIndex == 0) {
          // Set the pixel to transparent or a specific color to indicate background
          originalImage.setPixel(
              i, j, libImage.ColorFloat64.rgba(0, 0, 0, 0)); // Transparent
        }
      }
    }

    Uint8List outputImageBytes =
        Uint8List.fromList(libImage.encodePng(originalImage));

    setState(() {
      isProcessing = false;
      _outputImageBytes = outputImageBytes;
    });
  }

  Future<Uint8List> _loadNetworkImage(String urlImage) async {
    http.Response response = await http.get(
      Uri.parse(urlImage),
    );
    return response.bodyBytes;
  }

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _loadImage,
              child: const Text("Load Image"),
            ),
            if (_originalImageBytes != null)
              Image.memory(
                _originalImageBytes!,
                width: 200,
                height: 200,
              ),
            ElevatedButton(
              onPressed: _run,
              child: const Text("Process"),
            ),
            _buildResult(),
          ],
        ),
      ),
    );
  }

  Widget _buildResult() {
    print(isProcessing);
    print(_outputImageBytes);
    if (isProcessing || _outputImageBytes == null) {
      return const SizedBox(
        height: 200,
        width: 200,
        child: Center(
          child: CircularProgressIndicator(
            color: Colors.black,
          ),
        ),
      );
    }

    return Image.memory(
      _outputImageBytes!,
      width: 200,
      height: 200,
    );
  }

  Future<void> _loadModel() async {
    final options = InterpreterOptions();
    _interpreter = await Interpreter.fromAsset('assets/deeplabv3.tflite',
        options: options);
    _inputShape = _interpreter.getInputTensor(0).shape;
    _outputShape = _interpreter.getOutputTensor(0).shape;
  }
}

// OverlayPainter is used to draw mask on top of camera preview
class OverlayPainter extends CustomPainter {
  late final ui.Image image;

  updateImage(ui.Image image) {
    this.image = image;
  }

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(image, Offset.zero, Paint());
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
