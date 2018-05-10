#! /usr/bin/env python
# -*- coding:utf-8 -*- 

'''                                                                                                                                   
Created on Nov 8, 2015                                                                                                                
                                                                                                                                      
@author: duxin                                                                                                                     
'''

from ConfigOracleDBConnector import *
from ConfigDB import *
import cx_Oracle
import pandas as pd
from Utility import *
import time

'''
Class : OracleDBConnector
---------------------------------------------
Usage: This class is designed to query and insert(not implemented yet) data into oracle 
                data base. The connected oracle database could come from WanDe,Harvest saved
                Exchange database.
                Once an instance of this class is initiated, a connection parameter should be passed in containing the necessary 
                information including ip, port number, user name, password, etc for the connection request. This connection parameter 
                is a key which maps to another map, in which all connection parameters are specified(See ConfigOracleDBConnector.py) 

'''

class OracleDBConnector(object):
    
    '''
    Function: OracleDBConnector Class Constructor
    ------------------------------------------------------------------------
    Usage: Default parameter is 'wind' which maps to another map contains the information in the WanDe DB connection request.
    Current other options include 'exchange'.
    '''
    def __init__(self,dbKey=None):
        if dbKey is None:
            self.connPara = oracleConnParaMap['wind']
        else:
            self.connPara = oracleConnParaMap[dbKey]
        return
    
    
    '''
    Function: build_database_connection()
    ----------------------------------------------------------
    Usage: Build oracle database connection using the parameter passed in the constructor.
    '''
    def build_database_connection(self):
        try:                                           
            dsn=cx_Oracle.makedsn(self.connPara['host'],self.connPara['port'],self.connPara['database'])
            conn = cx_Oracle.connect(self.connPara['user'], self.connPara['password'],dsn)
        except cx_Oracle.DatabaseError  as e:
            print("Can not connect to server")
        return conn
    
    '''
    Function: get_query_stmt(tableName,colNames,constraints)
    ---------------------------------------------------
    Usage: Build oracle database query statement string.
                    Three Parameters: 
                    tableName: the queried table name
                    colNames: the names of the columns we wants to query
                    constraints: specify the constraints, in the data structure of  a map
    Note: This function is wrapped by get_data function
    '''
    def get_query_stmt(self,tableName,colNames,constraints=None):
        stmt = 'select '
        for col in colNames:
            stmt = stmt + col + ','
        stmt = stmt[0:len(stmt)-1] # remove the last comma                                                                           \                                                                                                                              
        stmt = stmt + ' from ' + oracleDBConf[tableName]
        if constraints is None:
                return stmt
        else:
                stmt = stmt + ' where '
                constraintStr = ''
                for curKey in constraints.keys():
                    constraintStr = constraintStr + ' and ' +curKey + constraints[curKey]
                constraintStr = constraintStr[5:len(constraintStr)]
                stmt = stmt + constraintStr
                return stmt
            
    '''
    Function: get_data(dbTableKey,colNames,constraints)
    ----------------------------------------------------------------------------------
    Usage: main member function to be called by instance to get the data frame requested.
                    oracleDBConf[dbTableKey] returns a table name that needs to be queried.
    '''
    '''
    def get_data(self,dbTableKey,colNames,constraints=None):
        assert colNames is not None
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(oracleDBConf[dbTableKey], colNames, constraints)
            cursor = conn.cursor()
            t0 = time.time()
            cursor.execute(stmt)
            df = pd.DataFrame.from_records(cursor.fetchall())
            if len(df)>0:
                df.columns = colNames
        except cx_Oracle.Error as e:
            conn.rollback()
            message = "Oracle Error %d: %s" % (e.args[0], e.args[1])
            print message
        finally:
            cursor.close()
            conn.close()
        print 'time elapsed for this oracle query: ',time.time()-t0
        return df
    '''
    def get_data(self,dbTableKey,colNames,constraints=None):
        assert colNames is not None
        df = pd.DataFrame(index=[],columns=colNames)
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(oracleDBConf[dbTableKey], colNames, constraints)
            print(stmt)
            # Check column tyoe
            
            cursor = conn.cursor()
            t0 = time.time()
            cursor.execute(stmt)
            r = list(cursor.fetchone())
            ind=[]
            for index,item in enumerate(r):
                if type(item) == cx_Oracle.LOB or type(item) == cx_Oracle.CLOB :
                    ind.append(index)
            cursor.close()
            
            # Get data
            cursor = conn.cursor()
            cursor.execute(stmt)
            if len(ind) == 0:
                df = pd.DataFrame.from_records(cursor.fetchall())
            else:
                for row in cursor:
                    row = list(row)
                    for i in ind:
                        if type(row[i]) == cx_Oracle.LOB or type(row[i]) == cx_Oracle.CLOB :
                            row[i]=row[i].read()
                    df0 = pd.DataFrame.from_records([row])
                    if len(df0)>0:
                        df0.columns = colNames
                    df = pd.concat([df,df0])
            if len(df)>0:
                df.columns = colNames
        except cx_Oracle.Error as e:
            conn.rollback()
            message = "Oracle Error %d: %s" % (e.args[0], e.args[1])
            print(message)
        finally:
            cursor.close()
            conn.close()
        print('time elapsed for this oracle query: ',time.time()-t0)
        return df
    
    '''
    Function: get_data_from_query(stmt,colNames)
    -------------------------------------------------------------------------------
    Usage: Auxiliary function that can be called directly in the case when the query statement is really.
    '''
    def get_data_from_query(self,stmt,colNames):
        conn = self.build_database_connection()
        cursor = conn.cursor()
        cursor.execute(stmt)
        queryData = cursor.fetchall()
        df = pd.DataFrame.from_records(queryData,columns=colNames)
        cursor.close()
        conn.close()
        return df

    '''
    Function: get_lob_data_from_query(stmt,colNames)
    -------------------------------------------------------------------------------
    Usage: Auxiliary function that can be called directly in the case when the query statement is really.
           The query columns include LOB objects.
    '''
    def get_lob_data_from_query(self,stmt,colNames):
        conn = self.build_database_connection()
        cursor = conn.cursor()
        cursor.execute(stmt)
        queryData = [[i.read().strip() if isinstance(i, cx_Oracle.LOB) else i for i in row] for row in cursor]
        df = pd.DataFrame.from_records(queryData,columns=colNames)
        cursor.close()
        conn.close()
        return df

if __name__ == '__main__':
    odc=OracleDBConnector('jydb')
    tablekey='jyLC_ExgIndustry'
    columns={'CompanyCode':'CompanyCode','FirstIndustryCode':'IndustryCode','FirstIndustryName':'IndustryName'}
    cons={'Standard':EQUAL('3'),'IfPerformed':EQUAL('1')}
    cols=columns.keys()
    newcols=columns.values()
    stmt=odc.get_query_stmt(tablekey,cols,cons)
    print(stmt)
    df=odc.get_data_from_query(stmt,newcols)
    df=df.astype(dtype={'CompanyCode':str,'IndustryCode':str})
    print(df['IndustryName'].unique())
    
    
