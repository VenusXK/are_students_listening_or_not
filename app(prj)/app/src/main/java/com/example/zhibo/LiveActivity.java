package com.example.zhibo;

import android.annotation.SuppressLint;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import io.vov.vitamio.MediaPlayer;
import io.vov.vitamio.Vitamio;
import io.vov.vitamio.widget.MediaController;
import io.vov.vitamio.widget.VideoView;


import java.net.MalformedURLException;
import java.net.URL;
import java.util.LinkedList;

import me.lake.librestreaming.core.listener.RESConnectionListener;
import me.lake.librestreaming.filter.hardvideofilter.BaseHardVideoFilter;
import me.lake.librestreaming.filter.hardvideofilter.HardVideoGroupFilter;
import me.lake.librestreaming.ws.StreamAVOption;
import me.lake.librestreaming.ws.StreamLiveCameraView;
import me.lake.librestreaming.ws.filter.hardfilter.GPUImageBeautyFilter;
import me.lake.librestreaming.ws.filter.hardfilter.extra.GPUImageCompatibleFilter;


public class LiveActivity extends AppCompatActivity {
    private static final String TAG = LiveActivity.class.getSimpleName();
    private StreamLiveCameraView mLiveCameraView;
    private String path = "";
    private VideoView mVideoView;
    //    private EditText mEditText;
    private Button mRTMPBACKBtn;
    //    private Button mStopBtn;
    private StreamAVOption streamAVOption;
    private LiveUI mLiveUI;
    private final String[] permissionNames = new String[]{
            android.Manifest.permission.RECORD_AUDIO,
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.CAMERA};

    private String rtmpUrl ="rtmp://192.168.137.1:1935/live/home";
    private String rtmpUrl_rec ="rtmp://192.168.137.1:1936/live/home";

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_function);
        final TextView surface_text = (TextView)findViewById(R.id.surface_text);
        final TextView data_nl = (TextView)findViewById(R.id.data_nl);
        final TextView data_l = (TextView)findViewById(R.id.data_l);

//        LineChart chart = (LineChart) findViewById(R.id.chart);
//        Vitamio.isInitialized(getApplication());
//        mEditText = (EditText) findViewById(R.id.url_b);

        mVideoView = (VideoView) findViewById(R.id.surface_view);
        mVideoView.setVisibility(View.GONE);


        final int[] click_time = {0};
        mRTMPBACKBtn = (Button) findViewById(R.id.btn_rtmpback);
        mRTMPBACKBtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View v){
                click_time[0] = click_time[0] + 1;
                if(click_time[0] == 1){
                    Toast.makeText(LiveActivity.this,"开始收流",Toast.LENGTH_LONG).show();
                }
                else if(click_time[0] >= 2 && click_time[0] <= 3){
                    Toast.makeText(LiveActivity.this,"正在收流 请耐心等待",Toast.LENGTH_LONG).show();
                }
                else{
                    Toast.makeText(LiveActivity.this,"您已操作此按键"+click_time[0]+"次\n若长时间未收流请清空后台重启应用",Toast.LENGTH_LONG).show();
                }
                make_Live();
                surface_text.setVisibility(v.GONE);
                mVideoView.setVisibility(v.VISIBLE);
                mRTMPBACKBtn.setText("收流中");
            }
        });

        initLiveConfig();
        mLiveUI = new LiveUI(LiveActivity.this,mLiveCameraView,rtmpUrl);

        final Handler handler = new Handler(){
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                surface_text.setText((String)msg.obj);
            }
        };

        final Handler handler2 = new Handler(){
            public void handleMessage(Message msg2) {
                super.handleMessage(msg2);
                data_nl.setText("未听课人数："+String.valueOf(msg2.arg1));
                data_l.setText("听课中人数："+String.valueOf(msg2.arg2));
            }
        };

        new Thread(){
            public void run(){
                int i = 0;
                while(true){
                    i++;
                    Message msg = Message.obtain();
                    if(i%4==0)
                        msg.obj = "请待pc端推流稳定后点<收流>按钮\n收流时间会较长 请耐心等待\n\n/";
                    if(i%4==1)
                        msg.obj = "请待pc端推流稳定后点<收流>按钮\n收流时间会较长 请耐心等待\n\n-";
                    if(i%4==3)
                        msg.obj = "请待pc端推流稳定后点<收流>按钮\n收流时间会较长 请耐心等待\n\n|";
                    if(i%4==2)
                        msg.obj = "请待pc端推流稳定后点<收流>按钮\n收流时间会较长 请耐心等待\n\n\\";
                    handler.sendMessage(msg);
                    try {
                        sleep(500);
                    }
                    catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();

        new Thread(){
            public void run(){
                while(true){
                    Message msg2 = Message.obtain();
                    DbUtil db = new DbUtil();
                    int[] res_arr;
                    try {
                        res_arr = db.search();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }

                    msg2.arg1 = res_arr[1];
                    msg2.arg2 = res_arr[0];
                    handler2.sendMessage(msg2);
                    try {
                        sleep(3000);
                    }
                    catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();

//        new Thread(){
//            public void run(){
//                chart.setBackgroundColor(Color.WHITE);
//                int order = 0;
//                while(true){
////                    Message msg3 = Message.obtain();
//                    make_chart(chart, order);
//                    chart.notifyDataSetChanged();
//                    order = order+1;
////                    handler2.sendMessage(msg3);
//                    try {
//                        sleep(3000);
//                    }
//                    catch (InterruptedException e) {
//                        e.printStackTrace();
//                    }
//                }
//            }
//        }.start();
    }

    public void make_Live(){
        mVideoView.setVideoPath(rtmpUrl_rec);
        mVideoView.setMediaController(new MediaController(this));
        mVideoView.requestFocus();
        mVideoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                mp.setPlaybackSpeed(1.0f);
            }
        });
    }

//    public void make_chart(LineChart chart, int order){
//        List<Entry> entries = new ArrayList<Entry>();

//        DbUtil db = new DbUtil();
//        int[] res_arr;
//        try {
//            res_arr = db.search_all();
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//        System.out.println(order);
//        for(int data = 0; data< order; data++) {
//            entries.add(new Entry(data, res_arr[data]));
//            entries.add(new Entry(data, 20+data));
//        }
//        System.out.println(entries);
//
//        LineDataSet dataSet = new LineDataSet(entries, "Label");
//
//        dataSet.setColor(Color.YELLOW);
//        dataSet.setValueTextColor(Color.YELLOW);
//
//        LineData lineData = new LineData(dataSet);
//        chart.setData(lineData);
//        chart.invalidate();


//        ArrayList<LineDataSet> lineDataSets = new ArrayList<LineDataSet>();
//        LineDataSet lineDataSet = new LineDataSet(yDataList, curveLable);

//        entries.clear();
//    }


    /**
     * 设置推流参数
     */

    public void initLiveConfig() {
        mLiveCameraView = (StreamLiveCameraView) findViewById(R.id.stream_previewView);

        //参数配置 start
        streamAVOption = new StreamAVOption();
        streamAVOption.streamUrl = rtmpUrl;
        //参数配置 end

        mLiveCameraView.init(this, streamAVOption);
        mLiveCameraView.addStreamStateListener(resConnectionListener);
        LinkedList<BaseHardVideoFilter> files = new LinkedList<>();
        files.add(new GPUImageCompatibleFilter(new GPUImageBeautyFilter()));
        //files.add(new WatermarkFilter(BitmapFactory.decodeResource(getResources(),R.mipmap.icon_login_logo),new Rect(100,100,200,200)));
        mLiveCameraView.setHardVideoFilter(new HardVideoGroupFilter(files));
    }

    RESConnectionListener resConnectionListener = new RESConnectionListener() {
        @Override
        public void onOpenConnectionResult(int result) {
            if(result == 0){
                String result_str = "成功";
                Toast.makeText(LiveActivity.this,"推流连接"+result_str+ "\n推流地址："+rtmpUrl,Toast.LENGTH_LONG).show();
            }
            else{
                String result_str = "失败";
                Toast.makeText(LiveActivity.this,"推流连接"+result_str+ "\n推流地址："+rtmpUrl,Toast.LENGTH_LONG).show();
            }
            //result 0成功  1 失败
        }

        @Override
        public void onWriteError(int errno) {
            Toast.makeText(LiveActivity.this,"推流出错,请尝试重连",Toast.LENGTH_LONG).show();
        }

        @Override
        public void onCloseConnectionResult(int result) {
            Toast.makeText(LiveActivity.this,"结束推流",Toast.LENGTH_LONG).show();
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mLiveCameraView.destroy();
    }
}
