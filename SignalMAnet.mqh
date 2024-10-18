//+------------------------------------------------------------------+
//|                                                     SignalMA.mqh |
//|                             Copyright 2000-2024, Borovikov.A |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//#include <betterfly.mqh>
#include <Expert\ExpertSignal.mqh>
#include <Math\Stat\Normal.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signals of indicator 'Moving Average'                      |
//| Type=SignalAdvanced                                              |
//| Name=Moving Average                                              |
//| ShortName=MA                                                     |
//| Class=CSignalMA                                                  |
//| Page=signal_ma                                                   |
//| Parameter=PeriodMA,int,12,Period of averaging                    |
//| Parameter=Shift,int,0,Time shift                                 |
//| Parameter=Method,ENUM_MA_METHOD,MODE_SMA,Method of averaging     |
//| Parameter=Applied,ENUM_APPLIED_PRICE,PRICE_CLOSE,Prices series   |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Class CSignalMA.                                                 |
//| Purpose: Class of generator of trade signals based on            |
//|          the 'Moving Average' indicator.                         |
//| Is derived from the CExpertSignal class.                         |
//+------------------------------------------------------------------+
#define EXPERT_MAGIC 123456
#define PATTERN_BARS 14

class CSignalnMA : public CExpertSignal
  {
protected:
   CiMA              m_ma;             // object-indicator
   //--- adjusted parameters
   
   int               m_ma_period;      // the "period of averaging" parameter of the indicator
   int               m_ma_shift;       // the "time shift" parameter of the indicator
   ENUM_MA_METHOD    m_ma_method;      // the "method of averaging" parameter of the indicator
   ENUM_APPLIED_PRICE m_ma_applied;    // the "object of averaging" parameter of the indicator
  
   int               iterations;                    // Количество итераций для обучения
   int               hidden_layer_size;             // Количество нейронов в скрытом слое
   double            ma_sell_values[PATTERN_BARS];             // Массив значений MA
   int               trend_sell_code[PATTERN_BARS];
   double            ma_buy_values[PATTERN_BARS];             // Массив значений MA
   int               trend_buy_code[PATTERN_BARS];
   
   double            data_n[];
   double            hidden_layer_output[];
   double            loss[];
   double            weights_hidden[4][2];    // Веса для скрытого слоя (4 нейрона, 2 входа)
   double            weights_output[4];       // Веса для выходного слоя (1 нейрон, 4 входа)
   double            biases_hidden[4];        // Биасы для скрытого слоя
   double            bias_output;             // Биас для выходного слоя
   int               m_extr_pos[];   // array of shifts of extremums (in bars)
   uint              m_extr_map;       // resulting bit-map of ratio of extremums of the oscillator and the price

   int               m_pattern_0;      // model 0 "price is on the necessary side from the indicator"
   int               m_pattern_1;      // model 1 "price crossed the indicator with opposite direction"
  

public:
                     CSignalnMA(void);
                    ~CSignalnMA(void);
   //--- methods of setting adjustable parameters
   void              PeriodMA(int value)                 { m_ma_period=value;          }
   void              Shift(int value)                    { m_ma_shift=value;           }
   void              Method(ENUM_MA_METHOD value)        { m_ma_method=value;          }
   void              Applied(ENUM_APPLIED_PRICE value)   { m_ma_applied=value;         }

   void              ClosePosition(void);
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual double       Direction(void);
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);
   


protected:
   //--- method of initialization of the indicator


   bool              InitnMA(CIndicators *indicators);
   //--- methods of getting data
   double            MA(int ind)                         { return(m_ma.Main(ind));     }
   double            DiffMA(int ind)                     { return(MA(ind+1)-MA(ind));   }
   double            OpenMA(int ind)                     { return(Open(ind));          }
   double            HighMA(int ind)                     { return(High(ind));          }
   double            LowMA(int ind)                      { return(Low(ind));           }
   double            CloseMA(int ind)                    { return(Close(ind));         }
   int               StateMain(int ind);
   
   void              CalculateMASellTrendCode(int bars);
   void              CalculateMABuyTrendCode(int bars);
   double            NeuralNetworkOutput();
   void              TrainNeuralNetwork(int bars, double rate, int &trend_code[]);
   double            ArraySigmoid(double &x[]);
   double            Sigmoid(double x);
   double            SigmoidDerivative(double x);
   double            RandomNormal();
   double            RandomRange(double min, double max);
   double            MinMaxNorm(double &inp[][]);
  
   
   


  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalnMA::CSignalnMA(void) : m_ma_period(14),
                               m_ma_shift(0),
                               m_ma_method(MODE_SMA),
                               m_ma_applied(PRICE_CLOSE),
                               iterations(10000),
                               hidden_layer_size(4)
{
 
 
 
 df.fun();
 
 ArrayResize(loss, iterations);
 ArrayResize(hidden_layer_output, hidden_layer_size);
   


  

//--- initialization of protected data
   m_used_series=USE_SERIES_OPEN+USE_SERIES_HIGH+USE_SERIES_LOW+USE_SERIES_CLOSE;
// Инициализируем массивы





}
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSignalnMA::~CSignalnMA(void)
  {
  }
//+------------------------------------------------------------------+
//| Validation settings protected data.                              |
//+------------------------------------------------------------------+
bool CSignalnMA::ValidationSettings(void)
  {
//--- validation settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data checks
   if(m_ma_period<=0)
     {
      printf(__FUNCTION__+": period MA must be greater than 0");
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| Create indicators.                                               |
//+------------------------------------------------------------------+
bool CSignalnMA::InitIndicators(CIndicators *indicators)
  {
//--- check pointer
   if(indicators==NULL)
      return(false);
//--- initialization of indicators and timeseries of additional filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- create and initialize MA indicator
   if(!InitnMA(indicators))
      return(false);
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//| Initialize MA indicators.                                        |
//+------------------------------------------------------------------+
bool CSignalnMA::InitnMA(CIndicators *indicators)
  {
//--- check pointer
   if(indicators==NULL)
      return(false);
//--- add object to collection
   if(!indicators.Add(GetPointer(m_ma)))
     {
      printf(__FUNCTION__+": error adding object");
      return(false);
     }




//--- initialize object
   if(!m_ma.Create(m_symbol.Name(),m_period,m_ma_period,m_ma_shift,m_ma_method,m_ma_applied))
     {
      printf(__FUNCTION__+": error initializing object");
      return(false);
     }
//--- ok
   return(true);
  }
//+------------------------------------------------------------------+
//|  Проверка состояния осциллятора.                                  |
//+------------------------------------------------------------------+
int CSignalnMA::StateMain(int ind)
  {
   int    res=0;
   double var;
//---
   for(int i=ind;;i++)
     {
      if(MA(i+1)==EMPTY_VALUE)
         break;
      var=MA(i);
      if(res>0)
        {
         if(var<0)
            break;
         res++;
         continue;
        }
      if(res<0)
        {
         if(var>0)
            break;
         res--;
         continue;
        }
      if(var>0)
         res++;
      if(var<0)
         res--;
     }
//---
   return(res);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


void CSignalnMA::ClosePosition(void)
{
   //--- объявление запроса и результата
   MqlTradeRequest request;
   MqlTradeResult  result;
   int total=PositionsTotal(); // количество открытых позиций   
   //--- перебор всех открытых позиций
   for(int i=total-1; i>=0; i--)
     {
      //--- параметры ордера
      ulong  position_ticket=PositionGetTicket(i);                                      // тикет позиции
      string position_symbol=PositionGetString(POSITION_SYMBOL);                        // символ 
      int    digits=(int)SymbolInfoInteger(position_symbol,SYMBOL_DIGITS);              // количество знаков после запятой
      ulong  magic=PositionGetInteger(POSITION_MAGIC);                                  // MagicNumber позиции
      double volume=PositionGetDouble(POSITION_VOLUME);                                 // объем позиции
      ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);    // тип позиции
      //--- вывод информации о позиции
      PrintFormat("#%I64u %s  %s  %.2f  %s [%I64d]",
                  position_ticket,
                  position_symbol,
                  EnumToString(type),
                  volume,
                  DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN),digits),
                  magic);
      //--- если MagicNumber совпадает
      if(magic==EXPERT_MAGIC)
        {
         //--- обнуление значений запроса и результата
         ZeroMemory(request);
         ZeroMemory(result);
         //--- установка параметров операции
         request.action   =TRADE_ACTION_DEAL;        // тип торговой операции
         request.position =position_ticket;          // тикет позиции
         request.symbol   =position_symbol;          // символ 
         request.volume   =volume;                   // объем позиции
         request.deviation=5;                        // допустимое отклонение от цены
         request.magic    =EXPERT_MAGIC;             // MagicNumber позиции
         //--- установка цены и типа ордера в зависимости от типа позиции
         if(type==POSITION_TYPE_BUY)
           {
            request.price=SymbolInfoDouble(position_symbol,SYMBOL_BID);
            request.type =ORDER_TYPE_SELL;
           }
         else
           {
            request.price=SymbolInfoDouble(position_symbol,SYMBOL_ASK);
            request.type =ORDER_TYPE_BUY;
           }
         //--- вывод информации о закрытии
         PrintFormat("Close #%I64d %s %s",position_ticket,position_symbol,EnumToString(type));
         //--- отправка запроса
         if(!OrderSend(request,result))
            PrintFormat("OrderSend error %d",GetLastError());  // если отправить запрос не удалось, вывести код ошибки
         //--- информация об операции   
         PrintFormat("retcode=%u  deal=%I64u  order=%I64u",result.retcode,result.deal,result.order);
         //---
        }
     }
   }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalnMA::Direction(void)
  {




   double result=m_weight*(LongCondition()-ShortCondition());

   Print("derection",result);
   Print("weight",m_weight);
   return(result);
  }
//+--
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalnMA::LongCondition(void)
  {
   int result=0;
   
   
   // Инициализируем веса и биасы случайными значениями
      for(int i = 0; i < hidden_layer_size; i++)
        {
         weights_hidden[i][0] = RandomNormal();  // Случайные веса для первого входа
         weights_hidden[i][1] = RandomNormal();  // Случайные веса для второго входа
         biases_hidden[i] = RandomRange(-1, 1);  // Биасы в диапазоне -1 до 1
        }

      bias_output = RandomRange(-1, 1);
      
      CalculateMABuyTrendCode(PATTERN_BARS);
      
      TrainNeuralNetwork(PATTERN_BARS,0.01,trend_buy_code);
   // Получаем входные данные для текущего бара (цены close и low)
      
   
   // Вычисляем результат работы нейросети для распознавания тренда
      ENUM_POSITION_TYPE type;
      double results = NeuralNetworkOutput();
      Print("ResultBuy=",results);
    
      
      
   //--- analyze positional relationship of the close price and the indicator at the first analyzed bar
      if(results > 0.99)
        {
   
         double res = results;
         result=(int(round(res)));
        }
      else
        {
         MqlTradeRequest request;
         type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(type==POSITION_TYPE_BUY)
           {
            if(results < 0.01)
              {
               ClosePosition();
               PrintFormat("Result=%g ",results);
              }
           }
         }

      //--- return the result
      Print("long",result); 
      
     
     return(result);
    }
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalnMA::ShortCondition(void)
  {
   int result=0;
   
   // Инициализируем веса и биасы случайными значениями
      for(int i = 0; i < hidden_layer_size; i++)
        {
         weights_hidden[i][0] = RandomNormal();  // Случайные веса для первого входа
         weights_hidden[i][1] = RandomNormal();  // Случайные веса для второго входа
         biases_hidden[i] = RandomRange(-1, 1);  // Биасы в диапазоне -1 до 1
        }
      
      bias_output = RandomRange(-1, 1);
      
      CalculateMASellTrendCode(PATTERN_BARS);
      
      TrainNeuralNetwork(PATTERN_BARS,1.0,trend_sell_code);
      // Получаем входные данные для текущего бара (цены close и low)
      
      // Вычисляем результат работы нейросети для распознавания тренда
      ENUM_POSITION_TYPE type;
      double results = NeuralNetworkOutput();
      Print("ResultShort",results);
      
      double open_pos=0;
      //--- analyze positional relationship of the close price and the indicator at the first analyzed bar
      if(results > 0.99)
        {
         double res = results;
         result=(int(round(res)));
        }
      else
        {
         MqlTradeRequest request;
         type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(type==POSITION_TYPE_SELL)
           {
            if(results < 0.01)
              {
               ClosePosition();
               PrintFormat("Result=%g ",results);
              }
           }
         }
   //--- return the result
      Print("Short",result);
      
    
   
   return(result);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Рассчитать скользящую среднюю и создать кодировку buy тренда          |
//+------------------------------------------------------------------+
void CSignalnMA::CalculateMABuyTrendCode(int bars)
     {
      int m_bars_count = bars;
     
      for(int i = 0; i < m_bars_count; i++)
        {
         ma_buy_values[i]=MA(i);
         // Определяем тренд: 1 для восходящего, 0 для нисходящего
         if(CloseMA(i) > MathMean(ma_buy_values))
            trend_buy_code[i] = 0; // Восходящий тренд
         else
            trend_buy_code[i] = 1; // Нисходящий тренд
         
        }
       Print("Point=",MathMean(ma_sell_values));
       ArrayPrint(trend_buy_code);
     }
//+------------------------------------------------------------------+
//| Рассчитать скользящую среднюю и создать кодировку sell тренда          |
//+------------------------------------------------------------------+
void CSignalnMA::CalculateMASellTrendCode(int bars)
  {
   int m_bars_count = bars;
  
   for(int i = 0; i < m_bars_count; i++)
     {
      ma_sell_values[i]=MA(i);
      // Определяем тренд: 1 для восходящего, 0 для нисходящего
      if(CloseMA(i) < MathMean(ma_sell_values))
         trend_sell_code[i] = 0; // Нисходящий тренд
      else
         trend_sell_code[i] = 1; // Восходящий тренд
      
     }
   Print("Point=",MathMean(ma_sell_values));
   ArrayPrint(trend_sell_code); 
  }

   
//+------------------------------------------------------------------+
//| Функция активации нейронной сети                                  |
//+------------------------------------------------------------------+
double CSignalnMA::NeuralNetworkOutput()
  {
   
   // Прямой проход через скрытый слой
   
   dir.df();
   for(int i=0; i<PATTERN_BARS; i++)
     {
      dir.inputs[i][0] = 0;
      dir.inputs[i][1] = 1;
     
     }
   //MinMaxNorm(dir.inputs);
   ArrayPrint(dir.inputs); 
   for(int i = 0; i < hidden_layer_size; i++)
     {
      
      hidden_layer_output[i] = Sigmoid(dir.inputs[i][0] * weights_hidden[i][0] +
                                       dir.inputs[i][1] * weights_hidden[i][1] +
                                       biases_hidden[i]);
     }

   // Прямой проход через выходной слой
   //double output = 0.0;
   for(int n = 0; n < 2; n++)
     {
      for(int i = 0; i < hidden_layer_size; i++)
        {
         df.output[n] += hidden_layer_output[i] * weights_output[i]+bias_output;
        }

     }

   // Возвращаем выходное значение с активацией

   return ArraySigmoid(df.output);
  }
//+------------------------------------------------------------------+
//| Функция обучения нейронной сети                                   |
//+------------------------------------------------------------------+
void CSignalnMA::TrainNeuralNetwork(int bars,double rate,int &trend_code[])
  {
   double m_learning_rate = rate;
   int m_bars_count = bars;
   dir.df();
   //for(int i=0; i<PATTERN_BARS; i++)
   //  {
   //   dir.inputs[i][0] = LowMA(i+1);
   //   dir.inputs[i][1] = HighMA(i+1);
     
   //  }
   //MinMaxNorm(dir.inputs); 
   for(int iter = 0; iter < iterations; iter++)
     {

      // Прямой проход для каждого бара
      for(int i = 0; i < m_bars_count; i++)
        {
        
         double target = trend_code[i];
         
         //ArrayPrint(dir.inputs);
         // Прямой проход через скрытый и выходной слой
         
         for(int j = 0; j < hidden_layer_size; j++)
           {
            hidden_layer_output[j] = Sigmoid(dir.inputs[i][0] * weights_hidden[j][0] +
                                             dir.inputs[i][1] * weights_hidden[j][1] +
                                             biases_hidden[j]);
           }

         double output = 0.0;
         for(int j = 0; j < hidden_layer_size; j++)
           {
            output += hidden_layer_output[j] * weights_output[j];
           }
         output += bias_output;
         output = Sigmoid(output);

         // Ошибка
         double error = target - output;
         loss[iter] = MathAbs(error);
         // Обратное распространение для выходного слоя
         double output_gradient = error * SigmoidDerivative(output);
         for(int j = 0; j < hidden_layer_size; j++)
           {
            weights_output[j] += m_learning_rate * output_gradient * hidden_layer_output[j];
           }
         bias_output += m_learning_rate * output_gradient;

         // Обратное распространение для скрытого слоя
         for(int j = 0; j < hidden_layer_size; j++)
           {
            double hidden_gradient = output_gradient * weights_output[j] * SigmoidDerivative(hidden_layer_output[j]);
            weights_hidden[j][0] += m_learning_rate * hidden_gradient * dir.inputs[i][0];
            weights_hidden[j][1] += m_learning_rate * hidden_gradient * dir.inputs[i][1];
            biases_hidden[j] += m_learning_rate * hidden_gradient;
           }
        }
      //PrintFormat("iter=%I64u loss=%g",iter,loss[iter]);
      //Sleep(500);
     }

  }
//+------------------------------------------------------------------+
//| Сигмоидная функция активации                                      |
//+------------------------------------------------------------------+
double CSignalnMA::Sigmoid(double x)
  {

   return 1.0 / (1.0 + MathExp(-x));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalnMA::ArraySigmoid(double &x[])
  {
   double y =0;
   for(int i=0; i<2; i++)
     {
      y = x[i];
      x[i]=1.0 / (1.0 + MathExp(-y));
     }
   return MathMean(x);
  }
//+------------------------------------------------------------------+
//| Производная сигмоидной функции                                    |
//+------------------------------------------------------------------+
double CSignalnMA::SigmoidDerivative(double x)
  {
   return x * (1.0 - x);
  }
//+------------------------------------------------------------------+
double CSignalnMA::RandomNormal()
  {

   return MathRandomNormal(0,1,hidden_layer_size,data_n);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalnMA::RandomRange(double min, double max)
  {
   if(min == max)
      return (min);
   double Min, Max;
   if(min > max)
     {
      Min = max;
      Max = min;
     }
   else
     {
      if(min < max)
        {
         Min = min;
         Max = max;
        }
      else
         return (min);
     }
   return (double(min + ((max - min) * (double)MathRand() / 32767.0)));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalnMA::MinMaxNorm(double &inp[][])
{
 
 for(int j=0; j<2; j++)
 {
    for(int i=0; i<PATTERN_BARS; i++)
    {
     dir.row_1[i] = inp[i][0];
     dir.row_2[i] = inp[i][1];
     double max[2] ={MathMax(dir.row_1), MathMax(dir.row_2)};
     double min[2] ={MathMin(dir.row_1),MathMin(dir.row_2)};
     
     dir.inputs[i][j] = (inp[i][j] - MathMin(min))/(MathMax(max)-MathMin(min));
    }
 }
return(0);
 
}
struct maxmin
{
 double row_1[PATTERN_BARS];
 double row_2[PATTERN_BARS];
 double inputs[PATTERN_BARS][2];
 long df();
};
maxmin dir;
long maxmin::df(void)
{
 for(int i=0; i<PATTERN_BARS; i++)   
 {
  
  dir.inputs[i][0] = 0;
  dir.inputs[i][1] = 0;
 }
 
 return(0);
}
struct arr
{
 double output[2];
 long       fun();

};
arr df;
long arr::fun()
{

 df.output[0]=0;
 df.output[1]=0;

return(0);
}
//+------------------------------------------------------------------+

