package com.example.sibi.fuzzy;

import android.content.Intent;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
public void browser1 (View view){
    Uri uri = Uri.parse("http://192.168.1.218:8080/get"); // missing 'http://' will cause crashed
    Intent intent = new Intent(Intent.ACTION_VIEW, uri);
    startActivity(intent);
}
    public void browser2 (View view){
        Uri uri = Uri.parse("http://192.168.1.218:8080/get"); // missing 'http://' will cause crashed
        Intent intent = new Intent(Intent.ACTION_VIEW, uri);
        startActivity(intent);
    }
}

