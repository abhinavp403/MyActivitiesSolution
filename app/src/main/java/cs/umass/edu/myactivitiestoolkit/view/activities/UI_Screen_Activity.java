package cs.umass.edu.myactivitiestoolkit.view.activities;

import android.os.Bundle;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Scanner;

import cs.umass.edu.myactivitiestoolkit.R;

public class UI_Screen_Activity extends AppCompatActivity {
    private String EVENT_DATE_TIME = "2018-12-31 10:30:00";
    private String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss";
    private LinearLayout linear_layout_1, linear_layout_2;
    private TextView tv_hour, tv_minute, tv_second;
    private Handler handler = new Handler();
    private Runnable runnable;
    private Button start_timer, stop_timer;
    private TextView txtServerRepCount;
    int walk= 0;
    int sit= 0;
    String[] w= new String[5];
    String[] s= new String[5];
    ArrayList<String> act= new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_option1);
        start_timer= (Button) findViewById(R.id.button);
        stop_timer= (Button) findViewById(R.id.button2);
        Log.d("Tag", "OnCreate called from UIScreen");
        //txtServerRepCount = findViewById(R.id.count);
        //String sessionId= getIntent().getStringExtra("EXTRA_SESSION_ID");
        //txtServerRepCount.setText(sessionId);
        //Intent in= getIntent();
        //String Email= in.getExtras().getString("Activity");
        //txtServerRepCount.setText(Email);
        act.add("sitting");
        act.add("sitting");
        act.add("walking");
        act.add("sitting");
        act.add("sitting");

        initUI();
        start_timer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                countDownStart();
            }
        });
        try {
            activity(act);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void initUI() {
        linear_layout_1 = (LinearLayout) findViewById(R.id.linear_layout_1);
        linear_layout_2 = (LinearLayout) findViewById(R.id.linear_layout_2);
        tv_hour = (TextView) findViewById(R.id.tv_hour);
        tv_minute = (TextView) findViewById(R.id.tv_minute);
        tv_second = (TextView) findViewById(R.id.tv_second);
    }

    private void countDownStart() {
        runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    handler.postDelayed(this, 1000);
                    SimpleDateFormat dateFormat = new SimpleDateFormat(DATE_FORMAT);
                    Date event_date = dateFormat.parse(EVENT_DATE_TIME);
                    Date current_date = new Date();
                    if (!current_date.after(event_date)) {
                        long diff = event_date.getTime() - current_date.getTime();
                        long Hours = 0;
                        long Minutes = 1;
                        long Seconds = 0;

                        tv_hour.setText(String.format("%02d", Hours));
                        tv_minute.setText(String.format("%02d", Minutes));
                        tv_second.setText(String.format("%02d", Seconds));
                    } else {
                        linear_layout_1.setVisibility(View.VISIBLE);
                        linear_layout_2.setVisibility(View.GONE);
                        handler.removeCallbacks(runnable);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        handler.postDelayed(runnable, 0);
    }

    protected void onStop() {
        super.onStop();
        handler.removeCallbacks(runnable);
    }

    private void activity(ArrayList act) throws FileNotFoundException {
        String contents = "";
        Scanner in = new Scanner(LandingPageActivity.file);
        while (in.hasNextLine())
        {
            contents += in.nextLine();
        }
        Log.d("filetag", contents);
//        for(Object s: act)
//        {   if(s.equals("walking"))
//            walk++;
//        else if(s.equals("sitting"))
//            sit++;
//        }
//        if(sit > walk) {
//            txtServerRepCount.setText("Sitting");
//        }
//        else {
//            txtServerRepCount.setText("Walking");
//        }
    }
}
