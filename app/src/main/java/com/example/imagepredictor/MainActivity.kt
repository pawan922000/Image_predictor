package com.example.imagepredictor

import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.imagepredictor.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {
    lateinit var selectb: Button
    lateinit var predictb: Button
    lateinit var result: TextView
    lateinit var imgview: ImageView
    lateinit var bitmap: Bitmap
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectb =findViewById(R.id.select)
        predictb= findViewById(R.id.predict)
        result= findViewById(R.id.result)
        imgview= findViewById(R.id.imageView)

        //img processor
        val imgprocessor= ImageProcessor.Builder()
            .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR)).build()

        val labels= application.assets.open("labels.txt").bufferedReader().readLines()

        selectb.setOnClickListener{
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent,100)

        }

        predictb.setOnClickListener{
            var tensorImage =TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)
            tensorImage= imgprocessor.process(tensorImage)

            val model = MobilenetV110224Quant.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(tensorImage.buffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxidx=0
            outputFeature0.forEachIndexed { index, fl ->
                if(outputFeature0[maxidx]<fl){
                    maxidx=index
                }
            }

            result.setText(labels[maxidx])

// Releases model resources if no longer used.
            model.close()

        }

    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode==100){
            val uri = data?.data
            bitmap= MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imgview.setImageBitmap(bitmap)
            
        }
    }
}