package controllers

import javax.inject._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.SparkSession
import play.api.mvc._

/**
 * This controller creates an `Action` to handle HTTP requests to the
 * application's home page.
 */
@Singleton
class HomeController @Inject() extends Controller{

  /**
   * Create an Action to render an HTML page with a welcome message.
   * The configuration in the `routes` file means that this method
   * will be called when the application receives a `GET` request with
   * a path of `/`.
   */

  def index = Action(parse.tolerantFormUrlEncoded) { request =>

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("My App")
      .config("spark.sql.warehouse.dir", "somevalue")
      .getOrCreate()

    val x = (request.body.get("xpos").map(_.head).get.toDouble) / 470
    val y = (request.body.get("ypos").map(_.head).get.toInt - 250).toDouble / 250
    val d_distance = request.body.get("defenderdistance").map(_.head).get.toInt
    val score_margin = request.body.get("scoremargin").map(_.head).get.toDouble / 30
    val period = request.body.get("period").map(_.head).get.toInt

    // Convert 1-based indices to 0-based.
    val d_distance_onehot = Vectors.sparse(7, Array(d_distance), Array(1))
    val period_onehot = Vectors.sparse(6, Array(period - 1), Array(1))

    val df_prediction = spark.createDataFrame(Seq(
      (x, y, d_distance_onehot, score_margin, period_onehot)
    )).toDF("x", "y", "d_distance_onehot", "score_margin", "period_onehot")

    //df_prediction.show

    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "d_distance_onehot", "score_margin", "period_onehot"))
      .setOutputCol("features")

    val output_1 = assembler.transform(df_prediction)

    val sameModel =RandomForestRegressionModel.read.load("rfModel_final")
    val predictions_test = sameModel.transform(output_1)
    val prediction = predictions_test.select("prediction").head().getDouble(0)
    println( "%.1f".format(prediction * 100 ) + "%")
    predictions_test.show

    val result = predictions_test.select("prediction").first().get(0).toString
    val resultpercent = (result.toDouble * 100).toInt.toString

    Ok(views.html.index(resultpercent))
  }
}
