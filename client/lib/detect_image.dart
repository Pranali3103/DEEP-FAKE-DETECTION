import 'package:blurrycontainer/blurrycontainer.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:provider/provider.dart';
import 'package:sc/api.dart';

class DetectImage extends StatefulWidget {
  const DetectImage({super.key});

  @override
  State<DetectImage> createState() => _DetectImageState();
}

class _DetectImageState extends State<DetectImage> {
  String cameraMode = "gallery";
  showInputSelector() {
    final api = Provider.of<API>(context, listen: false);
    return showDialog(
        context: context,
        builder: (builder) {
          return AlertDialog(
            title: const Text("Select Input"),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ListTile(
                  title: const Text("Gallery"),
                  onTap: () {
                    setState(() {
                      cameraMode = "gallery";
                      api.cameraMode = cameraMode;
                    });
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  title: const Text("Camera"),
                  onTap: () {
                    setState(() {
                      cameraMode = "camera";
                      api.cameraMode = cameraMode;
                    });
                    Navigator.pop(context);
                  },
                ),
              ],
            ),
          );
        });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<API>(
      builder: (BuildContext context, API value, Widget? child) {
        return Scaffold(
          body: value.predictedClass == ""
              ? Center(child: Lottie.asset("assets/animation.json"))
              : Container(
                  height: MediaQuery.of(context).size.height,
                  width: MediaQuery.of(context).size.width,
                  decoration: BoxDecoration(
                    color: value.predictedClass == "Real"
                        ? Colors.green
                        : value.predictedClass == "Fake"
                            ? Colors.red
                            : Colors.white,
                  ),
                  child: Container(
                    margin: const EdgeInsets.all(20.0),
                    width: MediaQuery.of(context).size.width,
                    height: MediaQuery.of(context).size.height,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(40.0),
                    ),
                    child: value.image == null
                        ? const Center(
                            child: Text('Lets identify real or fake!'))
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(40.0),
                            child: Image.file(
                              value.image!,
                              height: 150.0,
                              fit: BoxFit.cover,
                            ),
                          ),
                  ),
                ),
          floatingActionButtonLocation:
              FloatingActionButtonLocation.centerFloat,
          floatingActionButton: value.predictedClass == ""
              ? Container()
              : GestureDetector(
                  onLongPress: () {
                    showInputSelector();
                  },
                  child: BlurryContainer(
                    borderRadius: BorderRadius.circular(50.0),
                    child: FloatingActionButton.extended(
                      elevation: 0,
                      backgroundColor: Colors.white.withOpacity(.4),
                      label: cameraMode == "camera"
                          ? const Text("Camera Mode",
                              style: TextStyle(fontWeight: FontWeight.bold))
                          : const Text("Gallery Mode",
                              style: TextStyle(fontWeight: FontWeight.bold)),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(50.0),
                      ),
                      onPressed: value.getImage,
                    ),
                  ),
                ),
        );
      },
    );
  }
}
