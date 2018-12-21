package cs.umass.edu.myactivitiestoolkit.view.activities;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import cs.umass.edu.myactivitiestoolkit.R;

public class LandingPageActivity extends AppCompatActivity {
    private Button Option1, Option2;
    public static File file;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_landing_page);
        Option2 = (Button) findViewById(R.id.option2);
        Option1 = (Button) findViewById(R.id.option1);
        file = new File(getApplicationContext().getFilesDir(), "read");
        List<String> act = new ArrayList<String>();
        act.add("sitting");
        act.add("sitting");
        act.add("walking");
        act.add("sitting");
        act.add("sitting");
        Log.d("ENTER1", act.size()+"");
        String empty = "";
        /*if(act.size()==5) {
            Log.d("ifact", "bc");
            try {
                FileOutputStream stream = new FileOutputStream(LandingPageActivity.file, false);
                try{
                    for(int i= 0; i<5; i++) {
                        stream.write(empty.getBytes());
                    }
                } catch (IOException e) {
                    Log.d("Tag2", "ioexception");
                    e.printStackTrace();
                }
            } catch (FileNotFoundException e) {
                Log.d("Tag2", "ioexception");
                e.printStackTrace();
            }
        }*/

        String contents = "";
        Scanner in = null;
        try {
            in = new Scanner(LandingPageActivity.file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        while (in.hasNextLine())
        {
            contents += in.nextLine();
        }
        Log.d("filetag2", "Contents = " + contents);

        Option1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(LandingPageActivity.this, UI_Screen_Activity.class);
                startActivity(intent);
                finish();
            }
        });

        Option2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(LandingPageActivity.this, cs.umass.edu.myactivitiestoolkit.view.activities.MainActivity.class);
                startActivity(intent);
                finish();
            }
        });

    }
}
