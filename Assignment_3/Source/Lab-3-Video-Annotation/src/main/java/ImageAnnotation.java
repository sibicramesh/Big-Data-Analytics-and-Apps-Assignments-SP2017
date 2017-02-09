import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by sibi on 2/8/17.
 */

public class ImageAnnotation {
    public static void main(String[] args) throws IOException {
        final ClarifaiClient client = new ClarifaiBuilder("KKQIegBW9uOl_3vaMSzqq4QCfPNyNBvB7XNBz1vE", "xsY48eiDhhsFo5M7HE3F71ZYkB_tEQmemlWekTgG")
                .client(new OkHttpClient())
                .buildSync();
        client.getToken();

        File file = new File("output/mainframes");

        File[] files = file.listFiles();
        String[][] arr=new String[7][5];                                //2D Array Initialization
        for (int i=0; i<files.length;i++){
            ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(files[i]))
                    )
                    .executeSync();
            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();

            MBFImage image = ImageUtilities.readMBF(files[i]);
            int x = image.getWidth();
            int y = image.getHeight();

            List<Concept> data = predictions.get(0).data();

            for (int j = 0; j < data.size(); j++) {
                image.drawText(data.get(j).name(), (int)Math.floor(Math.random()*x), (int) Math.floor(Math.random()*y), HersheyFont.ROMAN_TRIPLEX, 18, RGBColour.ORANGE);
            }


            for(int k=0;k<=4;k++){
                arr[i][k]=data.get(k).name();                   // Storing Annotations in 2D Array
            }


            DisplayUtilities.displayName(image, "image" + i);
        }

        System.out.println("The video is focussed on nature with variety of flora and fauna.\nAt first the video starts off with "+arr[0][0]
                +", "+arr[0][1]+", "+arr[0][2]+", "+arr[0][3]+" and "+arr[0][4]+", then it continues with "+arr[1][2]+" which includes a "
                +arr[1][1]+" and they are collectively called as "+arr[1][0]+", followed by another "+arr[2][2]+" in "+arr[2][1]
                +". \nThe next frame is about " +arr[3][3]+" which includes "+arr[3][0]+" in "+arr[3][1]+" located in "+arr[3][4]+". The next frame is about" +
                " a "+arr[4][3]+" in a "+arr[4][0]+". \nfollowed by an "+arr[5][0]+" "+ arr[5][3]+". Finally an, "+arr[6][0]+" in "
                +arr[6][3]+".");

    }
}

