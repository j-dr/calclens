#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "profile.h"

#ifdef MEMWATCH
#include "memwatch.h"
#endif

/* global vars to use in profiling*/
static double GlobalProfileCurrTime[NUM_PROFILE_TAGS];
static double GlobalProfileTotTime[NUM_PROFILE_TAGS];
static double GlobalMinTimeBase;
static int isRunningProfileTag[NUM_PROFILE_TAGS];
static int ProfileInitFlag = 1;

#ifdef PROFILE_TIMESERIES
static double *GlobalProfileStartTime[NUM_PROFILE_TAGS];
static double *GlobalProfileStopTime[NUM_PROFILE_TAGS];
static long NGlobalProfileStartStopTimeBase[NUM_PROFILE_TAGS];
static long NGlobalProfileStartStopTime[NUM_PROFILE_TAGS];
#endif

void printStepTimesProfileTags(FILE *fp, long stepNum, const char *ProfileTagNames[])
{
  static int initPrevTimes = 1;
  static double prevTimes[NUM_PROFILE_TAGS];
  double currTimes[NUM_PROFILE_TAGS];
  double time;
  long i,j;
  int len,totlen=15;
  char stepNumName[] = "StepNum";
  char sc[100];
  
  if(initPrevTimes)
    {
      for(i=0;i<NUM_PROFILE_TAGS;++i)
	prevTimes[i] = 0.0;
      
      initPrevTimes = 0;
    }
  
  //make sure times have stopped
  time = MPI_Wtime();
  for(i=0;i<NUM_PROFILE_TAGS;++i) 
    {
      if(isRunningProfileTag[i])
        currTimes[i] = GlobalProfileTotTime[i] + time;
      else
        currTimes[i] = GlobalProfileTotTime[i];
    }
  
  //error check
  assert(fp != NULL);
  
  //print profile tag names
  if(ProfileTagNames != NULL)
    {
      len = strlen(stepNumName);
      fprintf(fp,"%s",stepNumName);
      if(totlen > len)
	for(j=0;j<totlen-len;++j)
	  fprintf(fp,"%s"," ");
      
      for(i=0;i<NUM_PROFILE_TAGS;++i)
	{
	  len = strlen(ProfileTagNames[i]);
	  fprintf(fp,"%s",ProfileTagNames[i]);
	  if(totlen > len)
	    for(j=0;j<totlen-len;++j)
	      fprintf(fp,"%s"," ");
	}
      fprintf(fp,"\n");
      fflush(fp);
      
      return;
    }
  
  //compute time for step
  sprintf(sc,"%ld",stepNum);
  len = strlen(sc);
  fprintf(fp,"%s",sc);
  if(totlen > len)
    for(j=0;j<totlen-len;++j)
      fprintf(fp,"%s"," ");
  for(i=0;i<NUM_PROFILE_TAGS;++i)
    fprintf(fp,"%g  ",currTimes[i] - prevTimes[i]);
  fprintf(fp,"\n");
  fflush(fp);
  
  //set prev time to curr time for next step
  for(i=0;i<NUM_PROFILE_TAGS;++i) 
    prevTimes[i] = currTimes[i];
}

double getTimeProfileTag(int tag)
{
  if(isRunningProfileTag[tag])
    return MPI_Wtime() + GlobalProfileCurrTime[tag];
  else 
    return GlobalProfileCurrTime[tag];
}

#ifdef PROFILE_TIMESERIES
double getTimeProfileTagSeries(int tag)
{
  if(isRunningProfileTag[tag])
    {
      return MPI_Wtime() - GlobalProfileStartTime[tag][NGlobalProfileStartStopTime[tag]];
    }
  else if(NGlobalProfileStartStopTime[tag] >= 1)
    {
      return GlobalProfileStopTime[tag][NGlobalProfileStartStopTime[tag]-1] - GlobalProfileStartTime[tag][NGlobalProfileStartStopTime[tag]-1];
    }
  else
    {
      return -1.0;
    }
}

double getPrevTimeProfileTagSeries(int tag, long NumPrev)
{
  long ind;
  
  if(isRunningProfileTag[tag])
    ind = NGlobalProfileStartStopTime[tag] - NumPrev;
  else
    ind = NGlobalProfileStartStopTime[tag] - NumPrev - 1;
  
  if(ind >= 0)
    return GlobalProfileStopTime[tag][ind] - GlobalProfileStartTime[tag][ind];
  else
    return -1.0;
}
#endif

double getTotTimeProfileTag(int tag)
{
  double time = 0.0;
  if(isRunningProfileTag[tag])
    time = MPI_Wtime();
  return GlobalProfileTotTime[tag] + time;
}

void logProfileTag(int tag)
{
  long i;
  double time;
#ifdef PROFILE_TIMESERIES
  double *tmp;
  const long NProfileAdd = 100;
#endif
  
  if(ProfileInitFlag)
    {
      for(i=0;i<NUM_PROFILE_TAGS;++i)
	{
	  isRunningProfileTag[i] = 0;
	  GlobalProfileTotTime[i] = 0.0;
	  GlobalProfileCurrTime[i] = 0.0;
	  
#ifdef PROFILE_TIMESERIES
	  NGlobalProfileStartStopTime[i] = 0;
	  NGlobalProfileStartStopTimeBase[i] = NProfileAdd;
	  GlobalProfileStartTime[i] = (double*)malloc(sizeof(double)*NProfileAdd);
	  assert(GlobalProfileStartTime[i] != NULL);
	  GlobalProfileStopTime[i] = (double*)malloc(sizeof(double)*NProfileAdd);
	  assert(GlobalProfileStopTime[i] != NULL);
#endif
	}
      GlobalMinTimeBase = MPI_Wtime();

      ProfileInitFlag = 0;
    }
  
  if(isRunningProfileTag[tag])
    {
      time = MPI_Wtime();
      GlobalProfileTotTime[tag] += time;
      GlobalProfileCurrTime[tag] += time;
#ifdef PROFILE_TIMESERIES
      GlobalProfileStopTime[tag][NGlobalProfileStartStopTime[tag]] = time;
      NGlobalProfileStartStopTime[tag] += 1;
#endif
      isRunningProfileTag[tag] = 0;
      
#ifdef PROFILE_TIMESERIES
      if(NGlobalProfileStartStopTime[tag] >= NGlobalProfileStartStopTimeBase[tag])
	{
	  tmp = (double*)realloc(GlobalProfileStartTime[tag],sizeof(double)*(NGlobalProfileStartStopTimeBase[tag]+NProfileAdd));
	  assert(tmp != NULL);
	  GlobalProfileStartTime[tag] = tmp;
	  
	  tmp = (double*)realloc(GlobalProfileStopTime[tag],sizeof(double)*(NGlobalProfileStartStopTimeBase[tag]+NProfileAdd));
	  assert(tmp != NULL);
	  GlobalProfileStopTime[tag] = tmp;
	  
	  NGlobalProfileStartStopTimeBase[tag] += NProfileAdd;
	}
#endif
    }
  else
    {
      time = MPI_Wtime();
      GlobalProfileTotTime[tag] -= time;
      GlobalProfileCurrTime[tag] = -time;
#ifdef PROFILE_TIMESERIES
      GlobalProfileStartTime[tag][NGlobalProfileStartStopTime[tag]] = time;
#endif
      isRunningProfileTag[tag] = 1;
    }
}

void printProfileInfo(const char name[], const char *ProfileTagNames[])
{
  /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
  MPI_Barrier(MPI_COMM_WORLD);
  /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
  
  /* get info about tasks */
  int NTasks,ThisTask;
  MPI_Comm_size(MPI_COMM_WORLD,&NTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  MPI_Status Stat;
  char fname[1000];
  
  double currProfileVec[NUM_PROFILE_TAGS];
  FILE *fp = NULL;
  long i,j,len;
  double Glbmin;
  
  const int totlen = 21;
  
  /* get global min time to set base*/
  MPI_Allreduce(&GlobalMinTimeBase,&Glbmin,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
  
#ifdef PROFILE_TIMESERIES
  /* print profile info*/
  sprintf(fname,"%s.series",name);
  remove(fname);
  for(len=0;len<NTasks;++len)
    {
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      MPI_Barrier(MPI_COMM_WORLD);
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      
      if(ThisTask == len)
	{
	  fp = fopen(fname,"a");
	  assert(fp != NULL);
	  for(i=0;i<NUM_PROFILE_TAGS;++i)
	    for(j=0;j<NGlobalProfileStartStopTime[i];++j)
	      fprintf(fp,"%d \t %ld \t %.10e \t %.10e \n",ThisTask,i,GlobalProfileStartTime[i][j]-Glbmin,GlobalProfileStopTime[i][j]-Glbmin);
	  fflush(fp);
	  fclose(fp);
	}
    }
#endif  

  /* print total times out to file name */
  for(j=0;j<NUM_PROFILE_TAGS;++j) //make sure times have stopped
    {
      if(isRunningProfileTag[j])
	currProfileVec[j] = GlobalProfileTotTime[j] + MPI_Wtime();
      else
	currProfileVec[j] = GlobalProfileTotTime[j];
    }
  
  sprintf(fname,"%s.tot",name);
  if(ThisTask == 0)
    {
      remove(fname);
      fp = fopen(fname,"w");
      assert(fp != NULL);
      
      if(ProfileTagNames != NULL)
	{
	  for(i=0;i<NUM_PROFILE_TAGS;++i)
	    {
	      len = strlen(ProfileTagNames[i]);
	      fprintf(fp,"%s",ProfileTagNames[i]);
	      if(totlen > len)
		for(j=0;j<totlen-len;++j)
		  fprintf(fp,"%s"," ");
	    }
	  fprintf(fp,"\n");
	}
      
      for(j=0;j<NUM_PROFILE_TAGS;++j)
	fprintf(fp,"%.10e     ",currProfileVec[j]);
      fprintf(fp,"\n");
    }
  
  for(i=1;i<NTasks;++i)
    {
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      MPI_Barrier(MPI_COMM_WORLD);
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
            
      if(ThisTask == i)
        MPI_Send(currProfileVec,NUM_PROFILE_TAGS,MPI_DOUBLE,0,(int) i,MPI_COMM_WORLD);
      else if(ThisTask == 0)
        MPI_Recv(currProfileVec,NUM_PROFILE_TAGS,MPI_DOUBLE,(int) i,(int) i,MPI_COMM_WORLD,&Stat);
      
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      MPI_Barrier(MPI_COMM_WORLD);
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
            
      if(ThisTask == 0)
        {
          for(j=0;j<NUM_PROFILE_TAGS;++j)
            fprintf(fp,"%.10e     ",currProfileVec[j]);
          fprintf(fp,"\n");
        }
    }
  
  if(ThisTask == 0)
    fclose(fp);
} 

void resetProfiler(void)
{
#ifdef PROFILE_TIMESERIES
  long i;
  for(i=0;i<NUM_PROFILE_TAGS;++i)
    {
      free(GlobalProfileStartTime[i]);
      free(GlobalProfileStopTime[i]);
    }
#endif

  ProfileInitFlag = 1;
}




