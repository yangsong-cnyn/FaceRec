/********************************************************
*Fraction.C by Yang Song
*
*Abstract data type representation of fraction
*
*Fraction is automatically simplified during construction
*
*Implement operators:
*    addition (+)
*    subtraction (-)
*    multiplication (*)
*    division (/)
*    equality (==)
*    non-equality (!=)
*    greater than (>)
*    less than (<)
*
*Also implement standard output: 
*i.e.  cout<<Fraction(-8,9);  ==> -8/9
*********************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

using namespace std;

class Fraction {
    int sign;
    int numer;
    int denom;

    static int GCD(int a, int b) {
        while(b)
        {
           int temp= b;
           b= a%b;
           a= temp;
        }
        return a;
    }

    static int LCM(int a,int b) {
        return a*b/GCD(a, b);
    }

    void simplify() {
        if (numer==0)
        {
           sign= 1;
           denom= 1;
           return;
        }

        int gcd= GCD(numer, denom);
        numer/= gcd;
        denom/= gcd;
    }
public:
    Fraction(int numer=0,int denom=1): numer(abs(numer)),denom(abs(denom)){
        if (denom==0)
        throw overflow_error("Division by zero");

        sign= ((numer>=0)^(denom>=0))? -1:1;
        simplify();
    }

    Fraction(const Fraction &fra): sign(fra.sign), numer(fra.numer), denom(fra.denom){};

    Fraction &operator= (const Fraction &fra) {
        sign= fra.sign;
        numer= fra.numer;
        denom= fra.denom;

        return *this;
    }

    friend ostream &operator<<(ostream &os, const Fraction& fra) {
        if (fra.numer==0)
        os<<"0";
        else
        os<<((fra.sign==1)?"":"-")<<fra.numer<<"/"<<fra.denom;

        return os;
    }

    //OPERATIONS
    Fraction operator+ (const Fraction& fra) const;
    Fraction operator- (const Fraction& fra) const;
    Fraction operator* (const Fraction& fra) const;
    Fraction operator/ (const Fraction& fra) const;

    //COMPARISONS
    bool operator==(const Fraction& fra) const;
    bool operator!=(const Fraction& fra) const;
    bool operator> (const Fraction& fra) const;
    bool operator< (const Fraction& fra) const;
};

//OPERATIONS
Fraction Fraction::operator+ (const Fraction& fra) const {
    int inumer= sign*numer*fra.denom+ fra.sign*fra.numer*denom;
    int idenom= fra.denom*denom;

    return Fraction(inumer, idenom);
}

Fraction Fraction::operator- (const Fraction& fra) const {
    int inumer= sign*numer*fra.denom- fra.sign*fra.numer*denom;
    int idenom= fra.denom*denom;

    return Fraction(inumer,idenom);
}

Fraction Fraction::operator* (const Fraction& fra) const {
    int inumer= sign*numer*fra.sign*fra.numer;
    int idenom= fra.denom*denom;

    return Fraction(inumer,idenom);
}

Fraction Fraction::operator/ (const Fraction& fra) const {
    int inumer= sign*numer*fra.sign*fra.denom;
    int idenom= denom*fra.numer;

    return Fraction(inumer,idenom);
}

//COMPARISONS
bool Fraction::operator==(const Fraction& fra) const{
    return (sign==fra.sign)&&(numer==fra.numer)&&(denom==fra.denom);
}

bool Fraction::operator!=(const Fraction& fra) const{
    return (sign!=fra.sign)||(numer!=fra.numer)||(denom!=fra.denom);
}

bool Fraction::operator> (const Fraction& fra) const{
    return (sign*numer*fra.denom> fra.sign*fra.numer*denom);
}

bool Fraction::operator< (const Fraction& fra) const{
    return (sign*numer*fra.denom< fra.sign*fra.numer*denom);
}

///////////////////////////////////////////////////////////////////////
//Test cases
int main() 
{
    try{
        Fraction A(-1, 12);
        Fraction B(2, 17);
    
        cout<<A<<" + "<<B<<" = "<<A+B<<endl;
        cout<<A<<" - "<<B<<" = "<<A-B<<endl;
        cout<<A<<" * "<<B<<" = "<<A*B<<endl;
        cout<<A<<" / "<<B<<" = "<<A/B<<endl;

        cout<<boolalpha<<(Fraction(4,-1)==Fraction(4,1))<<endl;
        cout<<boolalpha<<(Fraction(4,-1)!=Fraction(4,1))<<endl;
        cout<<boolalpha<<(Fraction(4,-1)<Fraction(4,1))<<endl;
        cout<<boolalpha<<(Fraction(4,-1)>Fraction(4,1))<<endl;

    
        cout<<Fraction(9,4)/Fraction(0,5)<<endl;
    }
    catch(overflow_error &e) {
        cout<<e.what()<<endl;
    }

    return 0;
}
