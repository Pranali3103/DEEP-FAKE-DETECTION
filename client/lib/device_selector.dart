import 'package:flutter/material.dart';
import 'package:sc/detect_image.dart';
import 'package:sc/detect_video.dart';

class DeviceDetector extends StatefulWidget {
  const DeviceDetector({super.key});

  @override
  State<DeviceDetector> createState() => _DeviceDetectorState();
}

class _DeviceDetectorState extends State<DeviceDetector> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Device Detector'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const DetectVideo(),
                  ),
                );
              },
              child: Container(
                height: 200,
                width: 200,
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Center(
                    child: Icon(
                  Icons.videocam,
                  color: Colors.white,
                  size: 100,
                )),
              ),
            ),
            const SizedBox(
              height: 20,
            ),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const DetectImage(),
                  ),
                );
              },
              child: Container(
                height: 200,
                width: 200,
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Center(
                  child: Icon(
                    Icons.image,
                    color: Colors.white,
                    size: 100,
                  ),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
