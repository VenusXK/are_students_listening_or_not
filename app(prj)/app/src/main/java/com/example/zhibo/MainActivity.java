package com.example.zhibo;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.tbruyelle.rxpermissions2.RxPermissions;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private final String[] permissionNames = new String[]{android.Manifest.permission.READ_PHONE_STATE,
            android.Manifest.permission.READ_EXTERNAL_STORAGE
            , android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            , android.Manifest.permission.CHANGE_NETWORK_STATE
            , android.Manifest.permission.CAMERA
            , android.Manifest.permission.RECORD_AUDIO
            , android.Manifest.permission.INTERNET
            , android.Manifest.permission.ACCESS_COARSE_LOCATION
            , android.Manifest.permission.ACCESS_FINE_LOCATION};

    private Button mPushBtn;
    private Button mPlayBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestPermissions();
    }

    /**
     * 查看权限是否已经全部授予
     */
    @SuppressLint("CheckResult")
    private void requestPermissions() {
        RxPermissions rxPermission = new RxPermissions(MainActivity.this);
        /***校验是否需要的权限均已授予***/
        rxPermission
                .request(permissionNames)
                .subscribe(new io.reactivex.functions.Consumer<Boolean>() {
                    @Override
                    public void accept(Boolean aBoolean) throws Exception {
                        if (aBoolean) {
                            initView();
                        } else {
                            // 用户拒绝了该权限，并且选中『不再询问』
                            Toast.makeText(MainActivity.this, "关闭的权限可以在手机设置中打开。",
                                    Toast.LENGTH_SHORT).show();
                            initView();
                        }
                    }
                });
    }

    /**
     * 初始化视图
     */
    private void initView() {
        mPushBtn = (Button) findViewById(R.id.push_stream_btn);
        mPushBtn.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.push_stream_btn:
                startActivity(new Intent(this, LiveActivity.class));
                break;
        }
    }


}
