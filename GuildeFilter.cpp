#include "GuideFilter.h"

static Mat boxfilter(const Mat & I, int r) {
	Mat result;         // 32x32 window�̴�.
	blur(I, result, Size(r, r)); 
	// ã�ƺô��� mean filter�̴�. 
	// rxr Ŀ�ΰ����� ���� ��� ����
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

		std::vector <Mat> pc; //�̰� ä���� �������� split�Ѵ�.
		split(p2, pc);

		for (std::size_t i = 0; i < pc.size(); ++i) 
			pc[i] = filterSingleChannel(pc[i]);
		
		merge(pc, result); // merge�Ѵ�.
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
	Mat mean_p  = boxfilter(p, r); // ���̵� �̹���, ������ radius�� r
	
	// 2            fmean(I.*p)
	Mat mean_Ip = boxfilter(I.mul(p), r);
	
	// I�� mul p �Ѱ�          meanI.*meanp
	Mat cov_Ip  = mean_Ip - mean_I.mul(mean_p);

	// E(ak,bk)= �ñ׸�wk�� ���ϴ� i((akIi + bk - pi)^2 + eak^2) // �̷��� ���� �ڽ�Ʈ ����� �̴ϸ����� �ϴ� ��

	// �̴ϸ����� ��Ų ��
	// a�� b�� coeffient�̴�. => �׷��� �ذ� �Ʒ���
	Mat a = cov_Ip / (var_I + eps); 
	Mat b = mean_p - a.mul(mean_I);  

	Mat mean_a = boxfilter(a, r);
	Mat mean_b = boxfilter(b, r);

	return mean_a.mul(I) + mean_b; 
	// qi = akIi + bk, ��� i�� wk�� ���Ѵ�.
	// q�� �ȼ� k�� �߽����� �ϴ� �����츦 ������ I�� �̿��ؼ� linear transform�� �Ͽ� �������
	// �̷��� local linear model�� I�� ������ ���� ���� ���� q�� ������ �����ٰ� ����
}

GuidedFilter::GuidedFilter(const Mat& I, int r, double eps) {
	if (I.channels() == 1)
		impl_ = new GuidedFilterMono(I, 2*r+1, eps); // 2*r+1 �����ش�. => window �� �� ���� ������� ����
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
	// ����þ� ���͸� ���� �ʴ´�.
	// ��ſ� window �ȿ��� mean filter�� ����.
	// ���̵� �̹����� ���� �ֱ� ������ ���귮�� ����. 
	Mat I, p;
	I = imread("cat.png", 0);
	double eps = 0.2 * 0.2; 
	int r = 16; // ������
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