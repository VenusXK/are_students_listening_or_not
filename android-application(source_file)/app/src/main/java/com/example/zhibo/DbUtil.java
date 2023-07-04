package com.example.zhibo;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class DbUtil {
    public static final String URL = "jdbc:mysql://192.168.137.1:3306/slsic";
    public static final String USER = "zjh";
    public static final String PASSWORD = "0000";

    public static int[] search() throws Exception {
        //1.加载驱动程序
        Class.forName("com.mysql.jdbc.Driver");
        //2. 获得数据库连接
        Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
        //3.操作数据库，实现增删改查
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("select * from temp where id = (select MAX(id) from temp);");
        //如果有数据，rs.next()返回true
        int arr[] = new int[5];
        int i = 0;
        while(rs.next()){
            arr[i++] = rs.getInt("data_l");
            arr[i++] = rs.getInt("data_nl");
        }
        for (int j = 0; j < i; j++){
            System.out.println(arr[j]);
        }
        return arr;
    }

    public static int[] search_all() throws Exception {
        //1.加载驱动程序
        Class.forName("com.mysql.jdbc.Driver");
        //2. 获得数据库连接
        Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
        //3.操作数据库，实现增删改查
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery("select * from temp order by id ASC");
        //如果有数据，rs.next()返回true
        int arr[] = new int[5];
        int i = 0;
        while(rs.next()){
            arr[i++] = rs.getInt("data_l");
            arr[i++] = rs.getInt("data_nl");
        }
        for (int j = 0; j < i; j++){
            System.out.println(arr[j]);
        }
        return arr;
    }
}
