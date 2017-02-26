import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;

import java.io.*;

public class ObjectFeatureExtraction {
    public static void main(String args[]) throws IOException {
        String inputFolder = "data/";
        String inputImage = "horse.png";
        String outputFolder = "output/";
        String[] IMAGE_CATEGORIES = { "horse", "bear", "elephant"};

        int input_class = 0;
        MBFImage mbfImage = ImageUtilities.readMBF(new File(inputFolder + inputImage));
        DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();
        LocalFeatureList<Keypoint> features = doGSIFTEngine.findFeatures(mbfImage.flatten());
        FileWriter fw = new FileWriter(outputFolder + IMAGE_CATEGORIES[0] + ".txt");
        BufferedWriter bw = new BufferedWriter(fw);
        for (int i = 0; i < features.size(); i++) {
            double c[] = features.get(i).getFeatureVector().asDoubleVector();
            bw.write(3 + ",");
            for (int j = 0; j < c.length; j++) {
                bw.write(c[j] + " ");
            }
            bw.newLine();
        }
        bw.close();
    }
}
