import org.json.JSONArray;
import org.json.JSONObject;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Created by Naga on 16-01-2017.
 */
@WebServlet(name = "webhook", urlPatterns = "/webhook")
public class TutorialDetails extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.setHeader("Access-Control-Allow-Origin", "*");
        resp.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE");
        resp.setHeader("Access-Control-Max-Age", "3600");
        resp.setHeader("Access-Control-Allow-Headers", "x-requested-with, X-Auth-Token, Content-Type");
        resp.setContentType("application/json");
        String topic = req.getParameter("topic");
        String msg = req.getParameter("msg");
        System.out.print(topic + " " + msg);
        resp.getWriter().write("URL Working");
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        StringBuilder buffer = new StringBuilder();
        BufferedReader reader = req.getReader();
        String line;
        while ((line = reader.readLine()) != null) {
            buffer.append(line);
        }
        String data = buffer.toString();
        System.out.println(data);
        String output = "";
        JSONObject params = new JSONObject(data);
        JSONObject result = params.getJSONObject("result");
        JSONObject parameters = result.getJSONObject("parameters");

            String frame = parameters.get("Frame").toString();
            String query = "https://api.mlab.com/api/1/databases/sample/collections/MySummary?q={%22name%22:%22" + frame + "%22}&apiKey=LAVIWeMcGlHcsBnaydvJPdeSL0ebwKQC";
            JSONObject jsonObject = getData(query);
            JSONObject js = new JSONObject();
            js.put("speech", jsonObject.get("action"));
            js.put("displayText", jsonObject.get("action"));
            js.put("source", "sibi database");
            output = js.toString();

        resp.setHeader("Content-type", "application/json");
        resp.getWriter().write(output);
    }

    public JSONObject getData(String query) throws IOException {
        URL obj = new URL(query);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();
        BufferedReader in = new BufferedReader(
                new InputStreamReader(con.getInputStream()));
        String inputLine;
        StringBuffer response = new StringBuffer();

        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();
        JSONArray jsonArray = new JSONArray(response.toString());
        JSONObject jsonObject = (JSONObject) jsonArray.get(0);
        return jsonObject;
    }
}
