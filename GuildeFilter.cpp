#include "GuideFilter.h"

static Mat boxfilter(const Mat & I, int r) {
	Mat result;         // 32x32 window이다.
	blur(I, result, Size(r, r)); 
	// 찾아봤더니 mean filter이다. 
	// rxr 커널가지고 범위 평균 낸것
	return result;
}

static Mat convertTo(const Mat&mat, int depth) {
	if (mat.depth() == depth) 
		return mat;
	
	Mat result;
	mat.convertTo(result, depth);
	return result;
}


class GuidedFilterImpl {
public: 
	virtual ~GuidedFilterImpl() {}
	Mat filter(const Mat &p, int depth);
protected:
	int Idepth;
private:
	virtual Mat filterSingleChannel(const Mat&p) const = 0;
};


class GuidedFilterMono : public GuidedFilterImpl {
public:
	GuidedFilterMono(const Mat&I, int r, double eps);

private:
	virtual Mat filterSingleChannel(const Mat&p) const;

private :
	int r;
	double eps;
	Mat I, mean_I, var_I;
};

Mat GuidedFilterImpl::filter(const Mat &p, int depth) {
	
	Mat p2 = convertTo(p, Idepth);
	Mat result;

	if (p.channels() == 1) {
		result = filterSingleChannel(p2);
	}
	else {

		std::vector <Mat> pc; //이게 채널이 여러개면 split한다.
		split(p2, pc);

		for (std::size_t i = 0; i < pc.size(); ++i) 
			pc[i] = filterSingleChannel(pc[i]);
		
		merge(pc, result); // merge한다.
	}

	return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const Mat&origI, int r, double eps) : r(r), eps(eps) {
	
	if (origI.depth() == CV_32F || origI.depth() == CV_64F)
		I = origI.clone();
	else
		I = convertTo(origI, CV_32F);
	
	Idepth = I.depth();

	mean_I = boxfilter(I, r);
	Mat mean_II = boxfilter(I.mul(I), r);
	var_I = mean_II - mean_I.mul(mean_I);

}
Mat GuidedFilterMono::filterSingleChannel(const Mat &p)const {

	// 1 
	Mat mean_p  = boxfilter(p, r); // 가이드 이미지, 윈도우 radius는 r
	
	// 2            fmean(I.*p)
	Mat mean_Ip = boxfilter(I.mul(p), r);
	
	// I와 mul p 한것          meanI.*meanp
	Mat cov_Ip  = mean_Ip - mean_I.mul(mean_p);

	// E(ak,bk)= 시그마wk에 속하는 i((akIi + bk - pi)^2 + eak^2) // 이렇게 만든 코스트 펑션을 미니마이즈 하는 것

	// 미니마이즈 시킨 해
	// a와 b는 coeffient이다. => 그래서 해가 아래것
	Mat a = cov_Ip / (var_I + eps); 
	Mat b = mean_p - a.mul(mean_I);  

	Mat mean_a = boxfilter(a, r);
	Mat mean_b = boxfilter(b, r);

	return mean_a.mul(I) + mean_b; 
	// qi = akIi + bk, 모든 i는 wk에 속한다.
	// q는 픽셀 k를 중심으로 하는 윈도우를 가지고 I를 이용해서 linear transform을 하여 만들어짐
	// 이러한 local linear model은 I가 엣지를 가질 때만 단지 q도 엣지를 가진다고 나옴
}

GuidedFilter::GuidedFilter(const Mat& I, int r, double eps) {
	if (I.channels() == 1)
		impl_ = new GuidedFilterMono(I, 2*r+1, eps); // 2*r+1 시켜준다. => window 의 한 변의 사이즈로 만듬
}

Mat guidedFilter(const Mat & I, const Mat & p, int r, double eps, int depth) {
	return GuidedFilter(I, r, eps).filter(p, depth);
}

GuidedFilter::~GuidedFilter() {
	delete impl_;
}

Mat GuidedFilter::filter(const Mat &p, int depth) const {
	return impl_->filter(p, depth);
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}
int main() {
	// 가우시안 필터를 쓰지 않는다.
	// 대신에 window 안에서 mean filter를 쓴다.
	// 가이드 이미지를 따로 넣기 때문에 연산량이 적다. 
	Mat I, p;
	I = imread("cat.png", 0);
	double eps = 0.2 * 0.2; 
	int r = 16; // 반지름
	eps *= 255 * 255;
    p = I;
	Mat q =guidedFilter(I ,p,  r, eps);
	imshow("origin", I);
	imshow("output", q);
	double  psnrV = getPSNR(I, q);
	cout << "psnrV: " << psnrV << endl;
	if (!I.data) { return -1; }
	waitKey(0);
	return 0;
}