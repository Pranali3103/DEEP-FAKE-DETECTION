import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:provider/provider.dart';
import 'package:sc/api.dart';

class DetectVideo extends StatefulWidget {
  const DetectVideo({super.key});

  @override
  State<DetectVideo> createState() => _DetectVideoState();
}

class _DetectVideoState extends State<DetectVideo> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detect Video'),
      ),
      body: Consumer<API>(
          builder: (BuildContext context, API value, Widget? child) {
        return value.isLoading
            ? Center(child: Lottie.asset("assets/processing.json"))
            : Column(
                children: [
                  Center(
                    child: ElevatedButton(
                      onPressed: () {
                        value.getVideo();
                      },
                      child: const Text('Upload Video'),
                    ),
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                      child: value.predictedClass == "Fake"
                          ? const Text(
                              "Fake Video",
                              style: TextStyle(
                                  color: Colors.red,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold),
                            )
                          : value.predictedClass == "Fake"
                              ? const Text(
                                  "Real Video",
                                  style: TextStyle(
                                      color: Colors.green,
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold),
                                )
                              : const Text(
                                  "",
                                  style: TextStyle(
                                      color: Colors.green,
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold),
                                ))
                ],
              );
      }),
    );
  }
}
