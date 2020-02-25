#include <QCoreApplication>
//#include "torch/csrc/api/include/torch/torch.h"
#include <iostream>

#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nms/cuda/nms_cuda.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string.h>
#include <cmath>
#include <time.h>
#include <map>
#include <memory.h>
#include <ctime>
#include <vector>
#include <cstdarg>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>

//#include <nms_cpu.hpp>
#define slots Q_SLOTS

using namespace std;

// Link Correelation_module
extern "C"
torch::Tensor correlation_cuda_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

int const threadsPerBlock = sizeof(unsigned long long) * 8;
// Initialize Anchor Box
torch::Tensor create_prior_boxes(){

        map<string,float> fmap_dims[2];

        fmap_dims[0]["conv4_3"]=80;

        fmap_dims[0]["conv6"]=40;
        fmap_dims[0]["conv7"]=20;
        fmap_dims[0]["conv8"]=10;
        fmap_dims[0]["conv9"]=10;
        fmap_dims[0]["conv10"]=10;

        fmap_dims[1]["conv4_3"]=64;

        fmap_dims[1]["conv6"]=32;
        fmap_dims[1]["conv7"]=16;
        fmap_dims[1]["conv8"]=8;
        fmap_dims[1]["conv9"]=8;
        fmap_dims[1]["conv10"]=8;

        map<string,float> scale_ratios[3];
        scale_ratios[0]["conv4_3"]=1.;
        scale_ratios[0]["conv6"]=1.;
        scale_ratios[0]["conv7"]=1.;
        scale_ratios[0]["conv8"]=1.;
        scale_ratios[0]["conv9"]=1.;
        scale_ratios[0]["conv10"]=1.;

        scale_ratios[1]["conv4_3"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv6"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv7"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv8"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv9"]=(float)pow(2,1/3.);
        scale_ratios[1]["conv10"]=(float)pow(2,1/3.);


        scale_ratios[2]["conv4_3"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv6"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv7"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv8"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv9"]=(float)pow(2,2/3.);
        scale_ratios[2]["conv10"]=(float)pow(2,2/3.);

        map<string, float> aspect_ratios[2];
        aspect_ratios[1]["conv4_3"]=1.;
        aspect_ratios[1]["conv6"]=1.;
        aspect_ratios[1]["conv7"]=1.;
        aspect_ratios[1]["conv8"]=1.;
        aspect_ratios[1]["conv9"]=1.;
        aspect_ratios[1]["conv10"]=1.;

        aspect_ratios[0]["conv4_3"]=(float)1/2;
        aspect_ratios[0]["conv6"]=(float)1/2;
        aspect_ratios[0]["conv7"]=(float)1/2;
        aspect_ratios[0]["conv8"]=(float)1/2;
        aspect_ratios[0]["conv9"]=(float)1/2;
        aspect_ratios[0]["conv10"]=(float)1/2;

        map<string, double> anchor_areas;

        anchor_areas["conv4_3"]=40*40.;
        anchor_areas["conv6"]=80*80.;
        anchor_areas["conv7"]=160*160.;
        anchor_areas["conv8"]=200*200.;
        anchor_areas["conv9"]=280*280.;
        anchor_areas["conv10"]=360*360.;
        string fmaps[6]={"conv4_3", "conv6", "conv7", "conv8", "conv9", "conv10"};
        double cx,cy;
        double h,w,anchor_h,anchor_w;
        string fmap_i;
        double prior_box[4];

        torch::Tensor prior_boxs = torch::rand({41760,4});

        int numbers=0;

        for(int fmap =0 ; fmap<6;fmap++){
            fmap_i=fmaps[fmap];

            for(int i=0;i<fmap_dims[1][fmap_i];i++){
                for(int j=0;j<fmap_dims[0][fmap_i];j++){
                   cx=(j+0.5)/fmap_dims[0][fmap_i];
                   cy=(i+0.5)/fmap_dims[1][fmap_i];
                   for(int s=0;s<1;s++){
                       for(int ar =0;ar<2;ar++){

                           h=sqrt(anchor_areas[fmap_i]/aspect_ratios[ar][fmap_i]);
                           w=aspect_ratios[ar][fmap_i]*h;
                           for( int sr =0;sr<3;sr++){
                               anchor_h=h*scale_ratios[sr][fmap_i]/512.;
                               anchor_w=w*scale_ratios[sr][fmap_i]/640.;
                               prior_boxs[numbers][0]=cx;
                               prior_boxs[numbers][1]=cy;
                               prior_boxs[numbers][2]=anchor_w;
                               prior_boxs[numbers][3]=anchor_h;
                               numbers++;

                           }
                       }
                   }
                }
            }
        }


        return prior_boxs;
}

torch::Tensor cxcy_to_xy(torch::Tensor cxcy){

    return torch::cat({cxcy.slice(1,0,2) - (cxcy.slice(1,2) / 2),cxcy.slice(1,0,2) + (cxcy.slice(1,2) / 2)}, 1);
}
torch::Tensor gcxgcy_to_cxcy(torch::Tensor gcxgcy,torch::Tensor priors_cxcy){
    torch::Tensor a,b,c;

    a=torch::mul(gcxgcy.slice(1,0,2),priors_cxcy.slice(1,2))/10+priors_cxcy.slice(1,0,2);
    b=torch::exp(gcxgcy.slice(1,2)/5)*priors_cxcy.slice(1,2);

    return torch::cat({a,b},1);
}
torch::Tensor find_intersection(torch::Tensor set_1,torch::Tensor set_2){
    torch::Tensor lower_bounds,upper_bounds,uml,intersection_dims;

    lower_bounds=torch::max(set_1.slice(1,0,2).unsqueeze(1),set_2.slice(1,0,2).unsqueeze(0));
    upper_bounds=torch::min(set_1.slice(1,2).unsqueeze(1),set_2.slice(1,2).unsqueeze(0));
    uml= (upper_bounds-lower_bounds);
    intersection_dims=uml.clamp(0);

     return intersection_dims.slice(2,0,1)*intersection_dims.slice(2,1,2);
}
torch::Tensor find_overlap(torch::Tensor set_1,torch::Tensor set_2){
    torch::Tensor intersection,areas_set_1,areas_set_2,union_;

    intersection=find_intersection(set_1,set_2);
    areas_set_1 = (set_1.slice(1,2,3) - set_1.slice(1,0,1)) * (set_1.slice(1,3,4)- set_1.slice(1,1,2)) ;

    areas_set_2 = (set_2.slice(1,2,3) - set_2.slice(1,0,1)) * (set_2.slice(1,3,4)- set_2.slice(1,1,2)) ;
    union_=areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection;

    return   intersection / union_;

}

// For Multiclass NMS
torch::Tensor detect_objects(torch::Tensor predicted_locs,torch::Tensor predicted_scores,torch::Tensor priors_xy,double min_score,double max_overlap,int top_k){
    torch::Tensor decode_loc,class_scores ,score_above_min_score;

    int classnum=predicted_scores.size(2);

    predicted_scores=predicted_scores.softmax(2);

    decode_loc=cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[0],priors_xy));

    int n_above_min_score;

    int classname=1;

    torch::Tensor out_boxes_total,out_scores_total;

    for(classname=1;classname<classnum;classname++){
        cout<<classname<<endl;
        class_scores=predicted_scores[0].slice(1,classname,classname+1);

        score_above_min_score=class_scores>min_score;
        n_above_min_score=score_above_min_score.sum().item<int>();

        if(n_above_min_score==0){
            torch::Tensor out=torch::zeros({1,5}).to(at::kCUDA);
            out[0][2]=1.;
            out[0][3]=1.;
            if(classname==1){
                out_boxes_total=out.slice(1,0,4);
                out_scores_total=out.slice(1,4,5);
            }
            continue;
        }

        torch::Tensor suppress;
        torch::Tensor out_boxes,out_scores;

        auto order_t=std::get<1>(class_scores.sort(0,true));
        auto class_sorted=class_scores.index_select(0,order_t.squeeze(-1));
        torch::Tensor decode_loc_sorted=decode_loc.index_select(0,order_t.squeeze(-1));


        class_sorted=class_sorted.slice(0,0,n_above_min_score);
        decode_loc_sorted=decode_loc_sorted.slice(0,0,n_above_min_score);



        auto overlap=find_overlap(decode_loc_sorted,decode_loc_sorted);
        //NMS
        suppress=torch::zeros((n_above_min_score)).to(at::kCUDA);


        for(int box=0;box<decode_loc_sorted.size(0);box++){



            if(suppress[box].item<int>()==1){
                continue;
            }

            torch::Tensor what=overlap[box]>max_overlap;
            what=what.to(at::kFloat);

            auto maxes=torch::max(suppress,what.squeeze(1));

            suppress=maxes;


            suppress[box]=0;

        }

        suppress=1-suppress;
        // in python out_boxes=decode_loc_sorted[suppress]
        auto indexing=torch::nonzero(suppress);

        out_boxes=decode_loc_sorted.index_select(0,indexing.squeeze(-1));

        out_scores=class_sorted.index_select(0,indexing.squeeze(-1));

        if(classname==1){
        out_boxes_total=out_boxes;
        out_scores_total=out_scores;
        }
        else{
            out_boxes_total=torch::cat({out_boxes_total,out_boxes},0);
            out_scores_total=torch::cat({out_scores_total,out_scores},0);
        }
    }
    return torch::cat({out_boxes_total,out_scores_total},1);
}

// string split
vector<string> split(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str);
    string temp;

    while (getline(ss, temp, delimiter)) {
        internal.push_back(temp);
    }

    return internal;
}

std::string format(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    #ifndef _MSC_VER
        size_t size = std::snprintf( nullptr, 0, format, args) + 1; // Extra space for '\0'
        std::unique_ptr<char[]> buf( new char[ size ] );
        std::vsnprintf( buf.get(), size, format, args);
        return std::string(buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    #else
        int size = _vscprintf(format, args);
        std::string result(++size, 0);
        vsnprintf_s((char*)result.data(), size, _TRUNCATE, format, args);
        return result;
    #endif
    va_end(args);
}
// Noramlize img
at::Tensor Normalize(at::Tensor img,int what){
    at::Tensor normimg,tensor_img_R_R,tensor_img_R_G,tensor_img_R_B;

    if(what==0){
        normimg=img.div_(255.).sub_(0.1598).div_(0.0813);
    }
    else{
        tensor_img_R_R=img.slice(1,0,1);
        tensor_img_R_G=img.slice(1,1,2);
        tensor_img_R_B=img.slice(1,2,3);
        tensor_img_R_R=tensor_img_R_R.div_(255.).sub_(0.3465).div_(0.2358);
        tensor_img_R_G=tensor_img_R_G.div_(255.).sub_(0.3219).div_(0.2265);
        tensor_img_R_B=tensor_img_R_B.div_(255.).sub_(0.2842).div_(0.2274);
        normimg=torch::cat({tensor_img_R_R,tensor_img_R_G,tensor_img_R_B},1);
    }
    return normimg;
}

int main(int argc, char *argv[])
{

    QCoreApplication a(argc, argv);
    // Load Image path txt
    std::ifstream file2("/media/rcvsejong/XavierSSD256/raid/datasets/kaist-rgbt/imageSets/test-all-20.txt");
       std::string str;
      std::string imgname_T,imgname_R;

    std::shared_ptr<torch::jit::script::Module> module,module_af;

   // SF  class
    //module = torch::jit::load("/home/rcvsejong/torch_test_v2/Halfway_plus.zip");
    // SF multi class
    module = torch::jit::load("/home/rcvsejong/torch_test_v3/SF+_C9.zip");
    // AF Before
    //module = torch::jit::load("/home/rcvsejong/torch_test_v2/AF_amp_before_v3.zip");
    // AF After
    //module_af = torch::jit::load("/home/rcvsejong/torch_test_v2/AF_amp_after_v2.zip");

    assert(module != nullptr);
     cv::Mat original_img,img_T,img_R;
     module->to(at::kCUDA);
     //module_af->to(at::kCUDA);
     string eval_path="/home/rcvsejong/torch_test_v2/eval_AF_2.txt";
     ofstream writefile(eval_path.data());
     // Using AF model using_af=1, not using using_af=0;
     int using_af=0;

     torch::Tensor priors_xy,detect_loc,detect_score;
     int i = 0;
     // Make Anchor Box
     priors_xy=create_prior_boxes();
     priors_xy=priors_xy.to(at::kCUDA);

     while (std::getline(file2, str))
     {
         vector<string> line_vector = split(str, '/');
         imgname_T="/media/rcvsejong/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/lwir/"+line_vector[2].substr(0,6)+".jpg";
         imgname_R="/media/rcvsejong/XavierSSD256/raid/datasets/kaist-rgbt/images/"+line_vector[0]+"/"+line_vector[1]+"/visible/"+line_vector[2].substr(0,6)+".jpg";

         //load
         img_T=cv::imread(imgname_T,cv::IMREAD_GRAYSCALE);
         img_R=cv::imread(imgname_R);
         original_img=img_R.clone();

         int width,height;
         width=img_T.size[1];

         height=img_T.size[0];

         ifstream file;
         // Image Processing

         ////////////////////Thermal
         cv::resize(img_T,img_T,cv::Size(640,512));
         at::Tensor tensor_img_T;

         std::vector<int64_t> dimsT{1,static_cast<int64_t>(img_T.channels()),
                                   static_cast<int64_t>(img_T.rows),
                                   static_cast<int64_t>(img_T.cols)};
         tensor_img_T=torch::from_blob(img_T.data,dimsT,at::kByte);

         tensor_img_T = tensor_img_T.to(at::kFloat);
         tensor_img_T=Normalize(tensor_img_T,0);
         ////////////////////////
         ///////////////////RGB

         cv::resize(img_R,img_R,cv::Size(640,512));
         //cout<<img_R.channels()<<endl;
         at::Tensor tensor_img_R;
         cv::cvtColor(img_R,img_R,cv::COLOR_BGR2RGB);
         std::vector<int64_t> dimsR{1,
                                   static_cast<int64_t>(img_R.rows),
                                   static_cast<int64_t>(img_R.cols),static_cast<int64_t>(img_R.channels())};

         tensor_img_R=torch::from_blob(img_R.data,dimsR,at::kByte);
         tensor_img_R=tensor_img_R.permute({0,3,1,2});
         tensor_img_R = tensor_img_R.to(at::kFloat);

         tensor_img_R=Normalize(tensor_img_R,1);

         /// ///////////////////
         std::vector<torch::jit::IValue> inputs;

         int64_t sizes2;
         tensor_img_R=tensor_img_R.to(at::kCUDA);
         tensor_img_T=tensor_img_T.to(at::kCUDA);
         // Make input vector
         inputs.push_back(tensor_img_R);
         inputs.push_back(tensor_img_T);




         auto original_dims = torch::tensor({width,height ,width,height}).unsqueeze(0);
         original_dims=original_dims.to(at::kFloat);

         original_dims=original_dims.to(at::kCUDA);

        torch ::Tensor input1,input2;
        int dilation_patch=2,padding=0,stride=1,patch_size=11,kernel_size=1;
        input1=torch::ones({1,64,128,160}).to(at::kCUDA);


        double t = (double)cv::getTickCount();
        std::vector<torch::jit::IValue> inputs_af;
      torch::Tensor loc,cls,score,_xvis,_xlwir,_yvis,_ylwir,_hxvis,_hxlwir;
      //cout<<inputs<<endl;

      auto output=module->forward(inputs).toTuple();
      // If want using AF_model
      if (using_af){


          auto locs=output->elements()[0];
          auto clss=output->elements()[1];
          auto visTuple=locs.toTuple();
          auto lwirTuple=clss.toTuple();
          _xvis=visTuple->elements()[0].toTensor();
          _yvis=visTuple->elements()[1].toTensor();
          _hxvis=visTuple->elements()[2].toTensor();

          _xlwir=lwirTuple->elements()[0].toTensor();
          _ylwir=lwirTuple->elements()[1].toTensor();
          _hxlwir=lwirTuple->elements()[2].toTensor();
          ///correlation
          auto cor_vis=correlation_cuda_forward(_xvis,_yvis,kernel_size,kernel_size,patch_size,patch_size,padding,padding,dilation_patch,dilation_patch,stride,stride);
          auto cor_lwir=correlation_cuda_forward(_xlwir,_ylwir,kernel_size,kernel_size,patch_size,patch_size,padding,padding,dilation_patch,dilation_patch,stride,stride);
          auto _xin=torch::cat({_xvis,_xlwir});
         auto _yin=torch::cat({_yvis,_ylwir});
         auto _hxin=torch::cat({_hxvis,_hxlwir});
         auto _corin=torch::cat({cor_vis,cor_lwir});



          inputs_af.push_back(_xin);
          inputs_af.push_back(_yin);
          inputs_af.push_back(_corin);
          inputs_af.push_back(_hxin);

          auto output_af=module_af->forward(inputs_af).toTuple();
           loc=output_af->elements()[0].toTensor();
          cls=output_af->elements()[1].toTensor();

      }
      else{ //SF
           loc=output->elements()[0].toTensor();
          cls=output->elements()[1].toTensor();
       }
      i++;

      // NMS
      detect_loc=detect_objects(loc,cls,priors_xy,0.1,0.45,200);
      detect_loc=detect_loc.to(at::kCUDA);
      detect_score=detect_loc.slice(1,4,5);

      detect_loc=detect_loc.slice(1,0,4);


      t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();

      //cout<<original_dims <<endl;
       detect_loc=detect_loc*original_dims;

      for(int box_num=0;box_num<detect_loc.size(0);box_num++){
          cout<<detect_loc[box_num]<<endl;
          cv::Rect point;
          float x1=detect_loc[box_num][0].item<float>();
          float y1=detect_loc[box_num][1].item<float>();
          float x2=detect_loc[box_num][2].item<float>();
          float y2=detect_loc[box_num][3].item<float>();
          //          cout<<x1<<endl;
          point.x=x1;
          point.y=y1;
          point.height=y2-y1;
          point.width=x2-x1;
//For Eval
//          if(writefile.is_open()){
//              writefile<<"score "<<detect_score[box_num].item<float>()<<" image_id "<<i<<" bbox "<<x1<<","<<y1<<","<<x2<<","<<y2<<"\n";

//          }
          //Draw Box on image
         cv::rectangle(original_img,point,(255,0,255));

      }
      std::cout <<"ID: "<<line_vector[2].substr(0,6)<< " Time : " << t << std::endl;
      char ii[100];
      sprintf(ii,"%06d.png",i);
      string savename="/media/rcvsejong/XavierSSD256/raid/test_2/";
      savename+=ii;
      //Save
      cv::imwrite(savename,original_img);


   }



   writefile.close();
   printf("Done");
   return a.exec();
}

